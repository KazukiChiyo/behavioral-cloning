import csv, cv2, sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.models import load_model

padding = "valid"
n_epoches = 100
strides = (1, 1)
batch_size = 32

initial_lr = 1e-3
lr_decay_factor = 0.75
lr_step_size = 8
min_lr = 1e-7
reduce_factor = 0.2

samples = []
with open('../dataset-new/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# reference: https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/b602658e-8a68-44e5-9f0b-dfa746a0cc1a
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../dataset-new/IMG/'+batch_sample[0].split('/')[-1]
                img = cv2.imread(name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                center_angle = float(batch_sample[3])
                images.append(img)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = load_model("model_cnn.h5")

def sched(e):
    return initial_lr*(lr_decay_factor**np.floor(e/lr_step_size))

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, verbose=1),
    LearningRateScheduler(schedule=sched, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=reduce_factor, patience=3, min_lr=min_lr, verbose=1)]

model.fit_generator(train_generator, steps_per_epoch=len(train_samples)//batch_size,
validation_data=validation_generator, validation_steps=len(validation_samples)//batch_size, epochs=n_epoches, callbacks=callbacks, verbose=1)
model.save("model_cnn_p2.h5")
