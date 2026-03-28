import os
import trimesh
import numpy as np
from shapely.geometry import Polygon


def finalize_mesh(mesh, target_vertices=128):
    """
    统一处理：确保水密、合并顶点、并确保顶点数达到阈值
    """
    mesh.merge_vertices()
    mesh.fill_holes()
    while len(mesh.vertices) < target_vertices:
        mesh = mesh.subdivide()
    
    mesh.process(validate=True)
    return mesh

def create_teardrop(radius=0.02, height=0.06):
    """
    基于球体变形的水滴：100%水密
    """
    # 使用较细致的球体作为基准 (UV球确保了极点闭合)
    mesh = trimesh.creation.uv_sphere(radius=radius, count=[32, 32])
    v = mesh.vertices
    # 归一化 Z 到 [0, 1]
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    z_norm = (v[:, 2] - z_min) / (z_max - z_min)
    
    # 水滴方程：顶部(z_norm=1)缩放为0，底部圆滑
    # 使用 sqrt(z) * (1-z) 产生流线型
    scale = 1.5 * np.sqrt(z_norm + 0.001) * (1.1 - z_norm)
    v[:, 0] *= scale
    v[:, 1] *= scale
    v[:, 2] = (z_norm - 0.5) * height # 缩放到目标高度并居中
    
    mesh.vertices = v
    return finalize_mesh(mesh)

def create_star_prism(outer_radius=0.025, inner_radius=0.012, height=0.05):
    """
    星形棱柱
    """
    n = 8 # 8角星
    angles = np.linspace(0, 2*np.pi, n*2, endpoint=False)
    radii = np.where(np.arange(n*2) % 2 == 0, outer_radius, inner_radius)
    points = np.column_stack([np.cos(angles)*radii, np.sin(angles)*radii])
    poly = Polygon(points)
    mesh = trimesh.creation.extrude_polygon(poly, height)
    mesh.apply_translation([0, 0, -height/2])
    return finalize_mesh(mesh)

def create_superellipsoid(size=0.04, power=0.5):
    """
    超椭球体：介于球体和方块之间的形状
    """
    mesh = trimesh.creation.box(extents=[size, size, size])
    # 通过多次细分和平滑模拟超椭球
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    trimesh.smoothing.filter_taubin(mesh, iterations=20)
    
    # 将顶点向中心归一化再膨胀，使其更圆润
    norms = np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
    mesh.vertices = (mesh.vertices / (norms + 1e-6)) * (size / 2)
    return finalize_mesh(mesh)

def create_random_rock(radius=0.02):
    """
    随机凸包（不规则碎石）
    """
    # 随机生成一些点，然后取凸包，凸包必然水密
    points = np.random.uniform(-radius, radius, (40, 3))
    mesh = trimesh.convex.convex_hull(points)
    return finalize_mesh(mesh)

def create_clover_column(radius=0.01, height=0.06):
    """
    三叶草型柱体
    """
    t = np.linspace(0, 2*np.pi, 100)
    # 三叶草极坐标方程: r = a + b*cos(3*theta)
    r = radius * (0.7 + 0.3 * np.cos(3 * t))
    points = np.column_stack([r * np.cos(t), r * np.sin(t)])
    poly = Polygon(points)
    mesh = trimesh.creation.extrude_polygon(poly, height)
    mesh.apply_translation([0, 0, -height/2])
    return finalize_mesh(mesh)

def create_solid_hourglass(height, radius, neck_radius=0.005, resolution=64):
    """生成密闭的平滑沙漏"""
    half_height = height / 2.0
    z = np.linspace(-half_height, half_height, 50)
    slope = (radius**2 - neck_radius**2) / (half_height**2)
    x = np.sqrt(z**2 * slope + neck_radius**2)
    curve_points = np.column_stack((x, z))
    # 包含中心点以确保 revolve 生成水密体
    full_profile = np.vstack(([[0, -half_height]], curve_points, [[0, half_height]]))
    mesh = trimesh.creation.revolve(full_profile, sections=resolution)
    return mesh


def create_twisted_prism(radius=0.02, height=0.06, twist_angle=np.pi/2, sides=6):
    """
    带有扭转效果的棱柱
    """
    # 创建基础多边形
    angles = np.linspace(0, 2*np.pi, sides, endpoint=False)
    points = np.column_stack([np.cos(angles)*radius, np.sin(angles)*radius])
    poly = Polygon(points)
    
    # 使用 extrude 及其 transform 参数实现扭转
    # 这里的关键是沿着 Z 轴步进并应用旋转
    mesh = trimesh.creation.extrude_polygon(poly, height)
    
    # 手动实现扭转：根据高度对顶点进行旋转
    v = mesh.vertices.copy()
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    z_dist = z_max - z_min
    for i in range(len(v)):
        # 计算当前顶点在高度上的比例
        fraction = (v[i, 2] - z_min) / z_dist
        angle = fraction * twist_angle
        # 旋转矩阵
        c, s = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[c, -s], [s, c]])
        v[i, :2] = np.dot(rot_matrix, v[i, :2])
    
    mesh.vertices = v
    mesh.apply_translation([0, 0, -height/2])
    return finalize_mesh(mesh)

def create_diamond(radius=0.02):
    # 正八面体的 6 个顶点
    vertices = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ]) * radius
    # 使用凸包生成面，确保 100% 水密
    mesh = trimesh.convex.convex_hull(vertices)
    return finalize_mesh(mesh)

def create_stepped_pyramid(base_size=0.04, top_size=0.01, height=0.05, num_steps=5):
    """
    创建方形阶梯金字塔
    base_size: 底层边长
    top_size: 顶层边长
    height: 总高度
    num_steps: 阶梯层数
    """
    step_height = height / num_steps
    # 计算每一层方块的边长 (线性递减)
    sizes = np.linspace(base_size, top_size, num_steps)
    
    meshes = []
    for i in range(num_steps):
        s = sizes[i]
        # 创建每一层的方块
        box = trimesh.creation.box(extents=[s, s, step_height])
        
        # 计算该层在 Z 轴的位置（从下往上堆叠）
        # -height/2 是底部，i * step_height 是当前层底部偏移，step_height/2 是方块中心偏移
        z_offset = -height/2 + i * step_height + step_height/2
        box.apply_translation([0, 0, z_offset])
        meshes.append(box)
    
    # 合并所有方块
    # concatenate 会保留内部面，finalize_mesh 中的 process 会尝试清理，
    # 但对于流场模拟，只要外表面闭合即可。
    pyramid = trimesh.util.concatenate(meshes)
    return finalize_mesh(pyramid)

def create_stepped_cone(base_radius=0.02, top_radius=0.005, height=0.04, num_steps=4):
    """
    创建圆形阶梯塔（由多个不同半径的圆柱堆叠而成）
    """
    step_height = height / num_steps
    radii = np.linspace(base_radius, top_radius, num_steps)
    
    meshes = []
    for i in range(num_steps):
        r = radii[i]
        # 创建每一层的圆柱
        cyl = trimesh.creation.cylinder(radius=r, height=step_height, sections=64)
        
        z_offset = -height/2 + i * step_height + step_height/2
        cyl.apply_translation([0, 0, z_offset])
        meshes.append(cyl)
        
    tower = trimesh.util.concatenate(meshes)
    return finalize_mesh(tower)


def create_simulation_obstacles():
    output_dir = "/data3/turbdiff/obstacle"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_meshes = {}

    # 基础几何体
    all_meshes["sphere"] = trimesh.creation.icosphere(subdivisions=4, radius=0.02)
    
    torus = trimesh.creation.torus(major_radius=0.03, minor_radius=0.01)
    torus2 = trimesh.creation.torus(major_radius=0.04, minor_radius=0.02)
    torus3 = trimesh.creation.torus(major_radius=0.03, minor_radius=0.02)
    r_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1])
    torus.apply_transform(r_matrix)
    torus2.apply_transform(r_matrix)
    torus3.apply_transform(r_matrix)
    all_meshes["torus"] = torus
    all_meshes["torus2"] = torus2
    all_meshes["torus3"] = torus3

    cylinder = trimesh.creation.cylinder(radius=0.02, height=0.06, sections=64)
    cylinder2 = trimesh.creation.cylinder(radius=0.02, height=0.07, sections=64)
    cylinder3 = trimesh.creation.cylinder(radius=0.02, height=0.08, sections=64)
    r_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
    cylinder.apply_transform(r_matrix)
    cylinder2.apply_transform(r_matrix)
    cylinder3.apply_transform(r_matrix)
    all_meshes["cylinder"] = cylinder
    all_meshes["cylinder2"] = cylinder2
    all_meshes["cylinder3"] = cylinder3

    cone = trimesh.creation.cone(radius=0.02, height=0.04, sections=64)
    cone.apply_transform(r_matrix)
    all_meshes["cone"] = cone

    cone2 = trimesh.creation.cone(radius=0.02, height=0.03, sections=64)
    cone2.apply_transform(r_matrix)
    all_meshes["cone2"] = cone2

    cone3 = trimesh.creation.cone(radius=0.02, height=0.02, sections=64)
    cone3.apply_transform(r_matrix)
    all_meshes["cone3"] = cone3

    capsule = trimesh.creation.capsule(radius=0.02, height=0.03)
    capsule.apply_transform(r_matrix)
    all_meshes["capsule"] = capsule

    capsule2 = trimesh.creation.capsule(radius=0.02, height=0.02)
    capsule2.apply_transform(r_matrix)
    all_meshes["capsule2"] = capsule2

    capsule3 = trimesh.creation.capsule(radius=0.02, height=0.04)
    capsule3.apply_transform(r_matrix)
    all_meshes["capsule3"] = capsule3

    # 进阶几何体
    ellipsoid = trimesh.creation.icosphere(subdivisions=3, radius=0.03)
    ellipsoid.apply_scale([1.0, 0.6, 0.3])
    all_meshes["ellipsoid"] = ellipsoid

    ellipsoid2 = trimesh.creation.icosphere(subdivisions=3, radius=0.03)
    ellipsoid2.apply_scale([1.2, 0.8, 0.4])
    all_meshes["ellipsoid2"] = ellipsoid2

    ellipsoid3 = trimesh.creation.icosphere(subdivisions=3, radius=0.03)
    ellipsoid3.apply_scale([0.8, 1.0, 0.3])
    all_meshes["ellipsoid3"] = ellipsoid3

    # 圆角方块
    box = trimesh.creation.box(extents=[0.04, 0.04, 0.04])
    rounded_box = box.subdivide().subdivide()
    trimesh.smoothing.filter_taubin(rounded_box, iterations=10)
    all_meshes["roundedbox"] = rounded_box

    # 弯曲形状 - 凸包一定是水密的
    angles = np.linspace(0, np.pi, 10)
    centers = np.c_[np.cos(angles)*0.03, np.sin(angles)*0.03, np.zeros_like(angles)]
    spheres = [trimesh.creation.uv_sphere(radius=0.01).apply_translation(c) for c in centers]
    all_meshes["bentshape"] = trimesh.util.concatenate(spheres).convex_hull

    # 沙漏
    hourglass = create_solid_hourglass(height=0.06, radius=0.02, neck_radius=0.005)
    hourglass.apply_transform(r_matrix)
    all_meshes["hourglass"] = hourglass

    # 1. 流线型水滴
    teardrop = create_teardrop(radius=0.02, height=0.06)
    teardrop.apply_transform(r_matrix)
    all_meshes["teardrop"] = teardrop
    
    # 2. 星形柱
    star_prism = create_star_prism(outer_radius=0.02, inner_radius=0.01, height=0.03)
    star_prism.apply_transform(r_matrix)
    all_meshes["starprism"] = star_prism

    # 3. 随机碎石
    all_meshes["rock"] = create_random_rock(radius=0.02)
    all_meshes["rock2"] = create_random_rock(radius=0.02)
    all_meshes["rock3"] = create_random_rock(radius=0.02)
    
    # 4. 圆角工业件 (Superellipsoid)
    all_meshes["roundedcube"] = create_superellipsoid(size=0.04)
    
    # 5. 三叶草异形柱
    clover_column = create_clover_column(radius=0.02, height=0.06)
    clover_column.apply_transform(r_matrix)
    all_meshes["clovercolumn"] = clover_column
    # 扭转六棱柱
    all_meshes["twistedhex"] = create_twisted_prism(radius=0.02, height=0.04, twist_angle=np.pi/2)
    all_meshes["twistedhex2"] = create_twisted_prism(radius=0.03, height=0.04, twist_angle=np.pi/2)
    # 钻石
    all_meshes["diamond"] = create_diamond(radius=0.02)

    steppedpyramid = create_stepped_pyramid(
        base_size=0.05, top_size=0.01, height=0.04, num_steps=5
    )
    r_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    steppedpyramid.apply_transform(r_matrix)
    all_meshes["steppedpyramid"] = steppedpyramid
    # 2. 圆形阶梯塔
    steppedcone = create_stepped_cone(
        base_radius=0.025, top_radius=0.005, height=0.04, num_steps=5
    )
    r_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    steppedcone.apply_transform(r_matrix)
    all_meshes["steppedcone"] = steppedcone


    
    # 统一验证与导出
    print(f"{'Geometry Name':<20} | {'Watertight':<10} | {'Faces':<8}")
    print("-" * 45)
    
    for name, mesh in all_meshes.items():
        # 最后的全局检查：确保没有重复顶点，法线统一
        mesh.fill_holes()
        mesh.process()
        
        file_path = os.path.join(output_dir, f"{name}.stl")
        mesh.export(file_path)
        
        water_status = "Yes" if mesh.is_watertight else "NO"
        print(f"{name:<20} | {water_status:<10} | {len(mesh.faces):<8}")

if __name__ == "__main__":
    create_simulation_obstacles()