import trimesh
import numpy as np

def generate_rays(scanner_position, scanner_config):
    """
    根据指定的扫描角度和射线数量生成射线方向。
    """
    scanner_position[2] += scanner_config['height_offset']
    vertical_angle_offset = np.radians(scanner_config['vertical_rotation'])
    horizontal_angle_offset = np.radians(scanner_config['horizontal_rotation'])
    num_vertical = int(scanner_config['vertical_angle'] / scanner_config['angle_resolution'])
    num_horizontal = int(scanner_config['horizontal_angle'] / scanner_config['angle_resolution'])
    vertical_angles = np.linspace(-scanner_config['vertical_angle'] / 2, scanner_config['vertical_angle'] / 2, num_vertical)
    horizontal_angles = np.linspace(-scanner_config['horizontal_angle'] / 2, scanner_config['horizontal_angle'] / 2, num_horizontal)
    x, y, z = [], [], []
    for v_angle in vertical_angles:
        v_angle_error = np.random.normal(0, scanner_config['angle_error_vertical'])
        adjusted_v_angle = np.radians(v_angle + v_angle_error) + vertical_angle_offset
        for h_angle in horizontal_angles:
            h_angle_error = np.random.normal(0, scanner_config['angle_error_horizontal'])
            adjusted_h_angle = np.radians(h_angle + h_angle_error) + horizontal_angle_offset
            x.append(np.cos(adjusted_v_angle) * np.cos(adjusted_h_angle))
            y.append(np.cos(adjusted_v_angle) * np.sin(adjusted_h_angle))
            z.append(np.sin(adjusted_v_angle))
    directions = np.vstack((x, y, z)).T
    ray_origins = np.tile(scanner_position, (len(directions), 1))
    return ray_origins, directions

def load_material_colors(material_file):
    colors = {}
    with open(material_file, 'r', encoding='utf-8') as file:
        current_material = None
        for line in file:
            line = line.strip()
            if line.startswith('newmtl'):
                current_material = line.split(' ', 1)[1]
            elif line.startswith('Kd') and current_material:
                kd_values = list(map(float, line.split()[1:4]))
                colors[current_material] = [int(255 * v) for v in kd_values]
    return colors

def parse_obj_info(obj_path):
    material_to_group = {}
    group_to_category = {}
    group_vertices = {}
    current_group = None
    with open(obj_path, 'r', encoding='utf-8') as file:
        vertices = []
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('g '):
                current_group = line.split()[1].strip()
                if current_group not in group_vertices:
                    group_vertices[current_group] = []
            elif line.startswith('usemtl ') and current_group:
                current_material = line.split()[1].strip()
                if current_material not in material_to_group:
                    material_to_group[current_material] = current_group
                    group_to_category[current_group] = get_label_from_group(current_group)
            elif line.startswith('f ') and current_group:
                face_vertices = [vertices[int(idx.split('/')[0]) - 1] for idx in line.strip().split()[1:]]
                group_vertices[current_group].extend(face_vertices)
    return material_to_group, group_to_category, group_vertices

def get_label_from_group(group_name):
    mapping = {
        'ibeam': 0, 'pipe': 1, 'pump': 2,
        'rbeam': 3, 'tank': 4, 'floor': 5
    }
    for key in mapping:
        if key in group_name.lower():
            return mapping[key]
    return 6

def simulate_laser_scanning(obj_path, material_path, scanner_config, scan_index):
    mesh = trimesh.load(obj_path, process=False)
    materials = load_material_colors(material_path)
    material_to_group, group_to_category, group_vertices = parse_obj_info(obj_path)
    scene = trimesh.Scene(mesh)

    if not scene.is_empty:
        bounds = scene.bounds
        print(f"OBJ文件的XYZ坐标范围: X({bounds[0][0]}, {bounds[1][0]}), Y({bounds[0][1]}, {bounds[1][1]}), Z({bounds[0][2]}, {bounds[1][2]})")

        if 'pump' in group_vertices:
            pump_vertices = np.array(group_vertices['pump'])
            min_z = np.min(pump_vertices[:, 2])
            max_z = np.max(pump_vertices[:, 2])
            print(f"pump构件的最低点Z坐标: {min_z}")
            print(f"pump构件的最高点Z坐标: {max_z}")
        else:
            print("未找到pump构件。")

        if 'scanner_position' in scanner_config:
            scanner_position = np.array(scanner_config['scanner_position'])
        else:
            center_point = scene.bounds.mean(axis=0)
            min_z = scene.bounds[0][2]
            scanner_position = np.array([center_point[0], center_point[1], min_z + scanner_config['height_offset']])

        ray_origins, directions = generate_rays(scanner_position, scanner_config)
        point_cloud = []

        for direction in directions:
            closest_point = None
            min_distance = np.inf
            ray_origin = scanner_position.reshape(1, -1)
            direction = direction.reshape(1, -1)

            for material_name, mesh_part in scene.geometry.items():
                hits, ray_indices, locations = mesh_part.ray.intersects_location(ray_origin, direction, multiple_hits=False)
                if hits.any():
                    distance = np.linalg.norm(hits[0] - scanner_position)
                    if distance < min_distance:
                        min_distance = distance
                        group_name = material_to_group.get(material_name, 'Default')
                        category = group_to_category.get(group_name, 7)
                        color = materials.get(material_name, [0, 0, 0])
                        closest_point = hits[0].tolist()

            if closest_point:
                point_cloud.append(closest_point)

        # 显示结果
        plot_scene(scene, point_cloud, scanner_position)

    else:
        print("场景为空，没有几何体被添加。")

def plot_scene(scene, point_cloud, scanner_position):
    # 创建激光点云
    if point_cloud:
        point_cloud = np.array(point_cloud)
        laser_points = trimesh.points.PointCloud(point_cloud, colors=[255, 0, 0])

    # 创建激光扫描仪位置的竖条
    scanner_bottom = scanner_position - np.array([0, 0, 10])
    scanner_line = trimesh.load_path([scanner_position, scanner_bottom])

    # 设置竖条颜色为绿色
    for entity in scanner_line.entities:
        entity.color = [0, 255, 0, 255]

    # 添加到场景
    scene.add_geometry(laser_points)
    scene.add_geometry(scanner_line)

    # 设置物体颜色为黑色
    for geometry in scene.geometry.values():
        if isinstance(geometry, trimesh.Trimesh):
            geometry.visual.face_colors = [0, 0, 0, 255]

    # 显示场景
    scene.show()

def main(sample_points, scanner_config, obj_path, material_path):
    for i, point in enumerate(sample_points):
        scanner_config['scanner_position'] = point
        simulate_laser_scanning(obj_path, material_path, scanner_config, i + 1)

# 样本点
sample_points = [[-7.63, -40.09, -12.518]]

scanner_config = {
    'vertical_angle': 90,
    'horizontal_angle': 360,
    'angle_resolution': 3,
    'max_distance': 100,
    'precision_error': 0.05,
    'height_offset': 2.5,
    'vertical_rotation': 0,
    'horizontal_rotation': 0,
    'angle_error_vertical': 0.01,
    'angle_error_horizontal': 0.01,
}

obj_path = 'obj_document/MEP_changjing2/xiangmu.obj'
material_path = 'obj_document/MEP_changjing2/xiangmu.mtl'

main(sample_points, scanner_config, obj_path, material_path)
