# 从 flask 模块导入所需的类和函数
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
# 从 sheet 模块导入 solve 函数
from sheet import solve

# 创建 Flask 应用实例
app = Flask(__name__)
# 修改上传文件夹为 static/uploads
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# 定义允许上传的文件扩展名
app.config['ALLOWED_EXTENSIONS'] = {'jpg'}

# 确保上传文件夹存在，如果不存在则创建
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    """
    检查文件名是否符合允许的扩展名规则。

    :param filename: 待检查的文件名
    :return: 如果文件名符合规则返回 True，否则返回 False
    """
    # 检查文件名中是否包含 . 且文件扩展名是否在允许的扩展名列表中
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def check(ans, IDAnswer):
    """
    对比学生答案和标准答案，计算正确题目数量和每道题的答题状态。

    :param ans: 标准答案列表
    :param IDAnswer: 学生答案列表，每个元素为 (题号, 答案) 元组
    :return: 正确题目数量和每道题的答题状态列表
    """
    # 初始化答题状态列表，索引 0 不使用，初始值为 'N' 表示没涂卡
    checkState = ['N'] * (len(ans) + 1)
    correctNum = 0
    # 遍历学生答案
    for i, a in IDAnswer:
        # 检查题号是否在有效范围内
        if 0 < i <= len(ans):
            # 如果学生答案和标准答案一致且该题之前未被标记过
            if a == ans[i - 1] and checkState[i] == 'N':
                checkState[i] = 'T'  # 标记为正确
            else:
                checkState[i] = 'F'  # 标记为错误
    # 统计正确题目数量
    for s in checkState[1:]:
        if s == 'T':
            correctNum += 1
    return correctNum, checkState

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    处理根路径的 GET 和 POST 请求，实现图片上传和答案校对功能。

    :return: 根据请求类型和处理结果返回相应的 HTML 页面
    """
    if request.method == 'POST':
        # 处理答题卡图片上传
        img_file = request.files.get('img_file')
        # 检查图片文件是否存在且文件名符合规则
        if img_file and allowed_file(img_file.filename):
            # 构建图片保存路径
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
            # 保存上传的图片
            img_file.save(img_path)
            # 调用 solve 函数处理图片，获取处理后的图片和相关信息
            (numImg, courseImg, ansImg), (NO, course, IDAnswer) = solve(img_path)
            # 构建学号图片保存路径
            num_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'numImg.jpg')
            # 构建科目图片保存路径
            course_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'courseImg.jpg')
            # 构建答案图片保存路径
            ans_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ansImg.jpg')
            # 保存处理后的学号图片
            cv2.imwrite(num_img_path, numImg)
            # 保存处理后的科目图片
            cv2.imwrite(course_img_path, courseImg)
            # 保存处理后的答案图片
            cv2.imwrite(ans_img_path, ansImg)

            # 处理标准答案文件上传
            ans_file = request.files.get('ans_file')
            ans = []
            # 检查标准答案文件是否存在且为 .txt 文件
            if ans_file and ans_file.filename.endswith('.txt'):
                # 读取标准答案文件内容并解码为字符串
                ans_content = ans_file.read().decode('utf-8')
                # 将标准答案内容按空格分割为列表
                ans = ans_content.split()

            if ans:
                # 调用 check 函数对比答案，获取正确题目数量和答题状态列表
                correctNum, checkState = check(ans, IDAnswer)
                result_text = ''
                result_text += f"科目：{course}\n"
                result_text += f"学号：{NO}\n"
                result_text += f"正确率：{correctNum}/{len(ans)}\n"
                result_text += "答题情况为（N 表示没涂卡，T 表示正确，F 表示错误）：\n"
                # 遍历答题状态列表，拼接答题情况信息
                for i in range(1, len(checkState)):
                    result_text += f"{i}: {checkState[i]}\n"
            else:
                result_text = "未上传标准答案文件，无法进行校对。"

            return render_template('result.html',
                                   # 使用 url_for 生成静态文件路径
                                   num_img_path=url_for('static', filename=f'uploads/{os.path.basename(num_img_path)}'),
                                   course_img_path=url_for('static', filename=f'uploads/{os.path.basename(course_img_path)}'),
                                   ans_img_path=url_for('static', filename=f'uploads/{os.path.basename(ans_img_path)}'),
                                   result_text=result_text)
        else:
            return render_template('error.html', error="请上传有效的 JPG 格式图片。")
    # 处理 GET 请求，返回首页
    return render_template('index.html')

if __name__ == '__main__':
    # 以调试模式运行 Flask 应用
    app.run(debug=True)
