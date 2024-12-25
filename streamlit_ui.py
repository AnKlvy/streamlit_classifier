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
MODEL_PATH = 'resnet18_cifar10_3.pth'  # Путь к сохраненной модели
BATCH_SIZE = 128  # Размер батча для обучения и тестирования
NUM_EPOCHS = 7  # Количество эпох для обучения
LEARNING_RATE = 0.001  # Скорость обучения

# Определим устройство: если доступен GPU, используем его, иначе - CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# Функция для получения загрузчиков данных для обучения и тестирования
def get_dataloaders():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Случайный горизонтальный переворот
        transforms.RandomRotation(10),  # Случайное вращение изображений
        transforms.Resize((32, 32)),  # Изменение размера изображений
        transforms.ToTensor(),  # Преобразование изображений в тензоры
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация изображений
    ])
    # Загружаем тренировочные и тестовые данные CIFAR-10
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Создаем загрузчики данных для обучения и тестирования
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    return trainloader, testloader

# Функция для загрузки модели
def load_model():
    model = models.resnet18()  # Загружаем предварительно обученную модель ResNet18
    model.fc = nn.Linear(model.fc.in_features, 10)  # Меняем последний слой для CIFAR-10 (10 классов)
    # model.to(device)  # Перемещаем модель на выбранное устройство (GPU/CPU)

    trained = False
    if os.path.exists(MODEL_PATH):  # Проверяем, существует ли файл с моделью
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))  # Загружаем веса модели
        print("Модель успешно загружена из файла.")
        trained = True
    else:
        print("Файл модели не найден. Будет использоваться новая модель.")
    return model, trained

# Функция для обучения модели
def train_model(model, trainloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Устанавливаем модель в режим обучения
        running_loss = 0.0
        for images, labels in trainloader:
            # images, labels = images.to(device), labels.to(device)  # Перемещаем данные на устройство
            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(images)  # Прогоняем изображения через модель
            loss = criterion(outputs, labels)  # Вычисляем ошибку
            loss.backward()  # Обратный проход для вычисления градиентов
            optimizer.step()  # Шаг оптимизации
            running_loss += loss.item()  # Аккумулируем потери
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader)}")
    torch.save(model.state_dict(), MODEL_PATH)  # Сохраняем модель после обучения
    print("Модель сохранена в файл.")

# Функция для оценки модели
def evaluate_model(model, testloader):
    model.eval()  # Устанавливаем модель в режим оценки
    correct = 0
    total = 0
    with torch.no_grad():  # Выключаем градиенты для ускорения
        for images, labels in testloader:
            # images, labels = images.to(device), labels.to(device)  # Перемещаем данные на устройство
            outputs = model(images)  # Прогоняем изображения через модель
            _, predicted = torch.max(outputs, 1)  # Получаем предсказанные классы
            total += labels.size(0)  # Общее количество примеров
            correct += (predicted == labels).sum().item()  # Количество правильных предсказаний
    accuracy = 100 * correct / total  # Вычисляем точность
    print(f"Accuracy on test set: {accuracy:.2f}%")

# Функция для предсказания класса для загруженного изображения
def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Изменяем размер изображения
        transforms.ToTensor(),  # Преобразуем изображение в тензор
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация изображения
    ])
    image = transform(image).unsqueeze(0)  # Преобразуем изображение и добавляем дополнительную размерность
    model.eval()  # Устанавливаем модель в режим оценки
    with torch.no_grad():  # Выключаем градиенты
        outputs = model(image)  # Прогоняем изображение через модель
        _, predicted = torch.max(outputs, 1)  # Получаем предсказанный класс
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # Классы CIFAR-10
    return classes[predicted.item()]  # Возвращаем метку предсказания

# Интерфейс Streamlit для загрузки изображения
file_up = st.file_uploader("Загрузите изображение (только JPG)", type="jpg")
if file_up is not None:
    image = Image.open(file_up)  # Открываем изображение
    st.image(image, caption='Загруженное изображение', use_container_width=True)  # Отображаем изображение
    st.write("Обрабатываю изображение...")

    # Загрузка модели и предсказание
    model, trained = load_model()  # Загружаем модель

    if not trained:  # Если модель не обучена, то обучаем ее
        st.write("### Обучение модели")
        trainloader, _ = get_dataloaders()  # Получаем данные для обучения
        criterion = nn.CrossEntropyLoss()  # Функция потерь
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Оптимизатор
        train_model(model, trainloader, criterion, optimizer, NUM_EPOCHS)  # Обучаем модель
    else:
        st.write("Модель уже обучена. Пропуск этапа обучения.")

    # Оценка модели
    st.write("### Оценка точности на тестовом наборе")
    _, testloader = get_dataloaders()  # Получаем данные для тестирования
    evaluate_model(model, testloader)  # Оцениваем модель

    # Предсказание
    st.write("### Предсказание для загруженного изображения")
    label = predict_image(model, image)  # Получаем предсказание для изображения
    st.write(f"Метка предсказания: **{label}**")  # Отображаем результат
