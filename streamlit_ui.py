from PIL import Image, ImageOps, ImageDraw
from torchvision import models, transforms
import torch
import streamlit as st

st.title("Классификация изображений с помощью PyTorch")
st.write("Загрузите изображение, чтобы получить предсказания на основе ResNet.")

file_up = st.file_uploader("Загрузите изображение (только JPG)", type="jpg")

def add_rounded_border(image, border_size=0, corner_radius=20, border_color=None):
    width, height = image.size
    mask = Image.new("L", (width + 2 * border_size, height + 2 * border_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        (0, 0, width + 2 * border_size, height + 2 * border_size),
        radius=corner_radius,
        fill=255
    )
    bordered_image = ImageOps.expand(image, border=border_size, fill=border_color)
    rounded_image = Image.new("RGBA", bordered_image.size)
    rounded_image.paste(bordered_image, mask=mask)
    return rounded_image

def predict(image):
    resnet = models.resnet101(pretrained=True)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    resnet.eval()
    out = resnet(batch_t)
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

if file_up is not None:
    image = Image.open(file_up)
    bordered_image = add_rounded_border(image, corner_radius=60)
    st.image(bordered_image, caption='Загруженное изображение', use_container_width=True)
    st.write("Обрабатываю изображение...")
    labels = predict(file_up)
    st.write("### Результаты классификации:")
    for i, (label, score) in enumerate(labels):
        label_after_comma = label.split(',')[1].strip() if ',' in label else label  # Сохраняем только часть после запятой
        st.write(f"**{i + 1}. {label_after_comma}** — вероятность: `{score:.2f}%`")
