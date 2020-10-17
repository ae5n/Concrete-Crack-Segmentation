from model import *
from data import *
from configs import *
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

np.random.seed(101)
tf.random.set_seed(101)

(train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_data(PATH)

print(f'Training Data: {len(train_images)}')
print(f'Validation Data: {len(val_images)}')
print(f'Testing Data: {len(test_images)}')

train_data = data_loader(image_paths=train_images, mask_paths=train_masks, image_size=IMAGE_SIZE, augment=True).tf_data(batch_size=BATCH_SIZE)
val_data = data_loader(image_paths=val_images, mask_paths=val_masks, image_size=IMAGE_SIZE, augment=False).tf_data(batch_size=BATCH_SIZE)
test_data = data_loader(image_paths=test_images, mask_paths=test_masks, image_size=IMAGE_SIZE, augment=False).tf_data(batch_size=BATCH_SIZE)

model, model_name = build_model(encoder='efficientnetb7', center='dac', full_skip=True, attention='sc', upscore='upall')
#model, model_name = unet()

model.summary()

MODEL = '/content/drive/Shared drives/CXRE01/Projects/Project Models/'+model_name+'.h5'

checkpoint = ModelCheckpoint(MODEL, 
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=NUM_EARLY_STOP,
                          verbose=1,
                          restore_best_weights=True)
reducelr = ReduceLROnPlateau(monitor='val_loss',
                             factor=0.1,
                             patience=NUM_UPDATE_LR,
                             verbose=1)
callbacks = [reducelr, earlystop, checkpoint]


train_steps = len(train_images) // BATCH_SIZE
val_steps = len(val_images) // BATCH_SIZE

if len(train_images) % BATCH_SIZE != 0:
    train_steps += 1
if len(val_images) % BATCH_SIZE != 0:
    val_steps += 1

## Training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=NUM_EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=callbacks)

plt.figure(figsize=(10, 8))
plt.title(model_name)
plt.grid(b=True, which='major', linestyle='-', alpha=0.7)
plt.grid(b=True, which='minor', linestyle=':', alpha=0.6)
plt.minorticks_on()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.gca().set_ylim([-0.03,1.03])
plt.legend()
plt.savefig(fname=model_name, dpi=300)

## Testing

test_steps = len(test_images) // BATCH_SIZE
if len(train_images) % BATCH_SIZE != 0:
    train_steps += 1
if len(val_images) % BATCH_SIZE != 0:
    val_steps += 1

model.evaluate(test_data, steps=test_steps)

## Show Predictions
def display(display_list):
    plt.figure(figsize=(15, 30))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.subplots_adjust(wspace=0.1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')

    plt.show()


def show_predictions(dataset, num=1):
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], pred_mask[0]])


show_predictions(test_data, 10)