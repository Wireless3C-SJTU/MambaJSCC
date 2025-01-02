'''
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
this script is used to split the image into small blocks to form the evaluation dataset of Kodak and CLIC2021
'''

from PIL import Image
import os



def split_image(image_path, block_size):
    """
    将输入图像分割为指定大小的不重叠小块。

    :param image_path: str, 输入图像的路径
    :param block_size: tuple, 分割块的大小 (宽度, 高度)
    :return: list, 分割的小图像列表
    """
    try:
        # 打开图像
        image = Image.open(image_path)
        image_width, image_height = image.size

        # 提取块大小
        block_width, block_height = block_size

        # 检查块大小的合法性
        if block_width <= 0 or block_height <= 0:
            raise ValueError("块大小必须为正整数。")

        # 创建结果列表
        blocks = []

        # 遍历图像以提取块
        for top in range(0, image_height, block_height):
            for left in range(0, image_width, block_width):
                # 确定块的区域
                box = (left, top, left + block_width, top + block_height)

                # 裁剪图像
                block = image.crop(box)
                
                # 如果块超出图像范围，则填充黑色背景
                if block.size != (block_width, block_height):
                    padded_block = Image.new("RGB", (block_width, block_height), (0, 0, 0))
                    padded_block.paste(block, (0, 0))
                    block = padded_block

                # 添加块到列表
                blocks.append(block)

        return blocks

    except Exception as e:
        print(f"处理图像时发生错误: {e}")
        return []

# 测试脚本
if __name__ == "__main__":
    # 输入图像路径和分块大小
    image_path = '/mnt/sda/datasets/Kodak/kodak/'
    block_width = 128
    block_height = 128
    for j, file in enumerate(os.listdir(image_path)):
        #print(file)
        if not os.path.exists(image_path):
            print("输入的图像路径无效！")
        else:
            # 调用函数分割图像
            blocks = split_image(image_path+file, (block_width, block_height))
            
            # 保存分割结果
            output_folder = "/mnt/sda/datasets/Kodak{}/kodak{}".format(block_width,block_width)
            #output_folder = "/mnt/sda/datasets/CLIC-{}/CLIC-{}".format(block_width,block_width)
            os.makedirs(output_folder, exist_ok=True)
            
            for i, block in enumerate(blocks):
                output_path = os.path.join(output_folder, f"{file[:-4]}_block_{i + 1}.png")
                block.save(output_path)
            
        print(f"第{j}图像分割完成！总共分割为 {len(blocks)} 个小块，结果保存在 '{output_folder}' 文件夹中。")
