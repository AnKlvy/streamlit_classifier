import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import streamlit as st
from PIL import Image

st.title("Классификация изображений с помощью ResNet18")
st.write("Загрузите изображение, чтобы получить предсказания на основе предварительно обученной модели ResNet18.")

# Конфигурация
MODEL_PATH = 'resnet18_cifar10_1.pth'
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 0.003

# Определим устройство: если доступен GPU, используем его, иначе - CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    return trainloader, testloader


def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)  # Перемещаем модель на выбранное устройство (GPU/CPU)

    trained = False
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        print("Модель успешно загружена из файла.")
        trained = True
    else:
        print("Файл модели не найден. Будет использоваться новая модель.")
    return model, trained


def train_model(model, trainloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)  # Перемещаем данные на устройство

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader):.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Модель сохранена в файл.")


def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)  # Перемещаем данные на устройство
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")


def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0).to(device)  # Перемещаем изображение на устройство
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return classes[predicted.item()]


file_up = st.file_uploader("Загрузите изображение (только JPG)", type="jpg")
if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Загруженное изображение', use_container_width=True)
    st.write("Обрабатываю изображение...")

    # Загрузка модели и предсказание
    model, trained = load_model()

    if not trained:
        st.write("### Обучение модели")
        trainloader, _ = get_dataloaders()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_model(model, trainloader, criterion, optimizer, NUM_EPOCHS)
    else:
        st.write("Модель уже обучена. Пропуск этапа обучения.")

    # Оценка модели
    st.write("### Оценка точности на тестовом наборе")
    _, testloader = get_dataloaders()
    evaluate_model(model, testloader)

    # Предсказание
    st.write("### Предсказание для загруженного изображения")
    label = predict_image(model, image)
    st.write(f"Метка предсказания: **{label}**")
