import numpy as np
import open3d as o3d
import cv2


def lightweight_depth_to_3d_fixed_v2(depth_path, output_path):
    print("1. 读取深度图...")
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError("无法读取深度图")

    print("2. 数据预处理...")
    depth = depth.astype(np.float32) / 1000.0  # 转换单位
    depth = cv2.medianBlur(depth, 5)  # 平滑深度图

    # 确保深度图有效
    if np.all(depth == 0):
        raise ValueError("深度图中无有效数据")

    # 缩小分辨率（如果过大）
    height, width = depth.shape
    if height > 1000 or width > 1000:
        depth = cv2.resize(depth, (width // 2, height // 2))

    # 创建相机内参
    fx = fy = width
    cx = width / 2
    cy = height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    print("3. 创建点云...")
    depth_o3d = o3d.geometry.Image(depth)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d,
        intrinsic,
        depth_scale=1.0,
        depth_trunc=5.0,
        stride=32  # 大步长减少点云密度
    )

    print("4. 点云后处理...")
    # 清理无效点
    valid_points = np.isfinite(np.asarray(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[valid_points])

    # 降采样（加大体素大小）
    pcd = pcd.voxel_down_sample(voxel_size=0.3)

    # 限制最大点数
    if len(pcd.points) > 10000:
        pcd = pcd.random_down_sample(0.1)  # 随机降采样

    print("5. 执行重建...")
    # 使用凸包重建作为替代方法
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_hull(pcd)
    except Exception as e:
        print("凸包重建失败，切换到Poisson重建")
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=3,  # 降低重建深度
            scale=1.1,
            linear_fit=True
        )

    print("6. 保存模型...")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"模型已保存至: {output_path}")

    return mesh


if __name__ == "__main__":
    try:
        depth_path = "/Users/dongxin/Downloads/aaa/aaa_depth.png"
        output_path = "/Users/dongxin/Downloads/aaa/output.obj"
        mesh = lightweight_depth_to_3d_fixed_v2(depth_path, output_path)
    except Exception as e:
        print(f"错误: {str(e)}")
