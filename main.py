import pygame
import pygame_gui
import Model
from transformers import AutoModel, AutoTokenizer
import torch
from pygame_gui.core import ObjectID

save_directory = "./phobert_model"
phobert = AutoModel.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = Model.PhoBERT_LSTM(tokenizer, phobert, 256, 1, 2)
model.load_state_dict(torch.load("Model/PhoBERT_BiLSTM_VNese.pth", map_location=torch.device('cpu')))

# Initialize Pygame
pygame.init()

# Set up the window
window_size = (500, 600)
window_surface = pygame.display.set_mode(window_size)
pygame.display.set_caption('Sentiment Analysis: Sentiment Analysis')

# Set up the GUI manager
manager = pygame_gui.UIManager(window_size, "theme.json")

# Create the UI components
input_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((100, 50), (150, 30)),
    text='Input text:',
    manager=manager
)

input_textbox = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect((200, 50), (200, 30)),
    manager=manager
)

action_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((197, 100), (100, 50)),
    text='Submit',
    manager=manager,
    object_id=ObjectID(class_id='@friendly_buttons',object_id='#hello_button')
)

result_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((100, 170), (50, 30)),
    text='Result: ',
    manager=manager
)
result = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((200, 170), (150, 30)),
    text='',
    manager=manager
)

image_path = 'image/default.jpg'  # Path to your image file
image_frame = pygame_gui.elements.UIImage(
    relative_rect=pygame.Rect((100, 250), (300, 300)),
    image_surface=pygame.image.load(image_path),
    manager=manager
)

def on_button_press():
    global model
    user_input = input_textbox.get_text()
    output = model([user_input])
    print(output)
    max_index = torch.argmax(output, dim=1).item()
    if max_index == 0:
        sentiment = "Negative"
        new_image_path = "image/negative.jpg"
    elif max_index == 1:
        sentiment = "Positive"
        new_image_path = "image/positive.jpg"
    result.set_text(f'{sentiment}')

    image_frame.set_image(pygame.image.load(new_image_path))
    image_frame.rebuild()

# Main loop
clock = pygame.time.Clock()
is_running = True

while is_running:
    time_delta = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == action_button:
                on_button_press()

        manager.process_events(event)

    manager.update(time_delta)
    window_surface.fill((255, 255, 255))
    manager.draw_ui(window_surface)

    pygame.display.update()

pygame.quit()
