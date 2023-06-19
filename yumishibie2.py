import streamlit as st
import os
from fastai.vision.all import *

# 获取当前文件所在的文件夹路径
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, "export.pkl")

# Load the model
learn_inf = load_learner(model_path)

# 定义数字标签到中文名称的映射关系
label_mapping = {
    0: "斑病",
    1: "锈病",
    2: "灰斑病",
    3: "健康"
}

st.title("玉米叶子健康判断")
st.write("请上传一张玉米叶子的照片，系统将自动识别并判定。")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If the user has uploaded an image
if uploaded_file is not None:
    # Display the image
    image = PILImage.create(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Get the predicted label
    pred, pred_idx, probs = learn_inf.predict(image)
    
    # 根据数字标签查找对应的中文名称
    predicted_label = label_mapping[pred_idx.item()]
    
    st.write(f"玉米叶子状态: {predicted_label}; 准确率: {probs[pred_idx.item()]:.04f}")
    
    # 添加条件语句以提供建议和提示
    if probs[pred_idx.item()] < 0.5:
        st.write("上传的图片可能不是玉米叶子，或者玉米叶子过多，请上传可能出现问题的一片叶子进行准确识别。")
    else:
        if pred_idx.item() == 0:
            st.write("1.选用抗病品种。2.病株残体，包括发病早期摘除下部病叶、收获季节田间秸秆和深耕灭茬。3.实行轮作，合理密植，适时早播，增施有机肥，提高栽培管理水平。4.发病初期及时采取药剂防治。可以轮换使用以下药剂：50％多菌灵可湿性粉剂500~600倍液、75％百菌清可湿性粉600~800倍液、65％代森锌可湿性粉400~500倍液，以及甲基硫菌灵(甲基托布津)、丙环唑、噁醚唑(世高)、咯菌清(适乐时、蓝宝石)等药剂。")
        elif pred_idx.item() == 1:
            st.write("田间发现零星病斑一定要及时拔出，同时进行喷药防治，一定要随时关注天气变化，雨后及时进行喷药防治，特别是在玉米大喇叭口期一定要提前喷药预防。此时喷药时一定要选择活性特别高，内吸性强的药剂进行喷药防治。在玉米大喇叭口期至吐丝期使用氟环唑、丙环唑、苯醚甲环唑、戊唑醇、嘧菌酯、吡唑醚菌酯、井冈霉素、噻呋酰胺、氯溴异氰尿酸等药剂进行喷雾防治。如果前期没有预防住，在锈病发病初期喷施，也可以控制锈病的发病程度，一定程度上减轻对产量的影响。")
        elif pred_idx.item() == 2:
            st.write("防治灰斑病首先要注意推广种植抗病品种，特别是兼抗几种叶斑病的优良品种。其次是在玉米收获后，清除田间的秸秆，耕翻灭茬。以减少来年的菌源。第三是合理密植、浇水和施肥，使植株健壮，提高抗病能力。最后是在灰斑病发病初期及时采用药剂防治。可选用50%多菌灵可湿性粉剂500倍液，或80%炭疽福美可湿性粉剂800倍液，或25％丙环唑10克/亩兑水45～60公斤喷雾，也可以使用甲基硫菌灵、福美甲胂、代森锰锌等药剂，间隔10天防治一次。")
        else:
            st.write("玉米叶子状态良好，没有发现病害。玉米要想获得高产，保护好叶片是关键。")


# 为第二个模型获取文件夹路径
model_path2 = os.path.join(path, "model.pkl")

# 加载第二个模型
learn_inf2 = load_learner(model_path2)


st.title("穿果蛾、瘿蝇、蝗虫和螟虫害虫识别")
st.write("请上传一张图片，系统将自动识别并判定。")

# 允许用户上传第二个图像
uploaded_file2 = st.file_uploader("为识别选择一张图片...", type=["jpg", "jpeg", "png"])

# 如果用户已经上传了第二个图像
if uploaded_file2 is not None:
    # 显示第二个图像
    image2 = PILImage.create(uploaded_file2)
    st.image(image2, caption="Uploaded Image for Module 2", use_column_width=True)
    
    # 获取第二个模型的预测标签
    pred2, pred_idx2, probs2 = learn_inf2.predict(image2)
    st.write(f"Prediction: {pred2}; Probability: {probs2[pred_idx2]:.04f}")

    