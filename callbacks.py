# Purpose: Define all the callbacks used during training (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def get_callbacks():
    # Early stopping callback
    early_stop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    
    # Learning rate scheduler callback
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", patience=3, factor=0.5, min_lr=0.000001)
    
    # Model checkpoint callback
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    
    return [early_stop, reduce_lr, checkpoint]
