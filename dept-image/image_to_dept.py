import requests
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import base64
import json


class LocalDepthEstimator:
    def __init__(self, api_token, max_retries=5):
        """
        初始化深度估计器
        api_token: Hugging Face API token
        max_retries: 最大重试次数
        """
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.api_url = "https://api-inference.huggingface.co/models/Intel/dpt-hybrid-midas"
        self.max_retries = max_retries

    def call_api_with_retry(self, image_data):
        """
        调用API并处理重试逻辑
        """
        for attempt in range(self.max_retries):
            try:
                print(f"尝试第 {attempt + 1} 次请求...")
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    data=image_data
                )

                # 检查是否是模型加载状态
                if response.status_code == 503:
                    result = response.json()
                    if "estimated_time" in result:
                        wait_time = result["estimated_time"]
                        print(f"模型正在加载，预计等待时间: {wait_time:.1f} 秒")
                        time.sleep(wait_time + 1)  # 等待预估时间并添加1秒缓冲
                        continue

                return response

            except Exception as e:
                print(f"请求出错: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

        return None

    def process_image(self, input_path, output_dir):
        """
        处理单张图片
        input_path: 输入图片的路径
        output_dir: 输出目录的路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 读取图片
        try:
            with open(input_path, "rb") as f:
                image_data = f.read()
        except FileNotFoundError:
            print(f"文件未找到: {input_path}")
            return None, None

        # 调用API
        print("正在调用API处理图片...")
        response = self.call_api_with_retry(image_data)

        if response is None:
            print("达到最大重试次数，处理失败")
            return None, None

        if response.status_code == 200:
            try:
                # 解析JSON响应
                result = response.json()
                if "depth" not in result:
                    print("API返回的数据格式不正确")
                    return None, None

                # 解码base64图像数据
                image_data = base64.b64decode(result["depth"])
                image = Image.open(io.BytesIO(image_data))
                depth_map = np.array(image)

                # 生成文件名（使用原始文件名+时间戳）
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 保存原始深度图
                raw_filename = os.path.join(output_dir, f"{base_name}_depth_{timestamp}.png")
                cv2.imwrite(raw_filename, depth_map)
                print(f"原始深度图已保存: {raw_filename}")

                # 创建并保存彩色可视化版本
                normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_PLASMA)
                colored_filename = os.path.join(output_dir, f"{base_name}_depth_colored_{timestamp}.png")
                cv2.imwrite(colored_filename, colored)
                print(f"彩色深度图已保存: {colored_filename}")

                return raw_filename, colored_filename

            except json.JSONDecodeError:
                print("API返回的响应不是有效的JSON格式")
                return None, None
            except UnidentifiedImageError as e:
                print(f"无法识别深度图像格式。错误: {str(e)}")
                return None, None
            except Exception as e:
                print(f"处理深度图时发生错误: {str(e)}")
                return None, None
        else:
            print(f"API调用失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None, None

    def process_directory(self, input_dir, output_dir, extensions=('.jpg', '.jpeg', '.png')):
        """
        处理整个目录的图片
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        extensions: 支持的图片格式
        """
        results = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(extensions):
                input_path = os.path.join(input_dir, filename)
                print(f"\n处理图片: {filename}")
                raw, colored = self.process_image(input_path, output_dir)
                if raw and colored:
                    results.append({
                        'input': filename,
                        'raw_depth': raw,
                        'colored_depth': colored
                    })
        return results


def display_image(image_path):
    """
    显示图像的辅助函数，使用matplotlib代替cv2.imshow
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()


def main():
    # 配置参数
    API_TOKEN = "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"  # 替换为你的API Token

    # 设置输入输出路径
    INPUT_IMAGE = "/Users/dongxin/Downloads/aaa/aaa.png"  # 替换为你的输入图片路径
    OUTPUT_DIR = "/Users/dongxin/Downloads/aaa"  # 替换为你想要保存结果的目录路径

    # 初始化深度估计器
    estimator = LocalDepthEstimator(API_TOKEN, max_retries=10)  # 设置最大重试次数为10

    # 处理单张图片
    print("开始处理单张图片...")
    raw_path, colored_path = estimator.process_image(INPUT_IMAGE, OUTPUT_DIR)

    if raw_path and colored_path:
        print("\n处理完成!")
        print(f"原始深度图保存在: {raw_path}")
        print(f"彩色深度图保存在: {colored_path}")

        # 显示结果
        display_image(colored_path)


if __name__ == "__main__":
    main()
