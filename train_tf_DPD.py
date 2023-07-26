#####################################################################
#################### CONTENT OF train_tf_DPD.py #####################
#####################################################################

import tensorflow as tf

# toggle flag to run Horovod
run_hvd = False
if run_hvd:
    import horovod.tensorflow as dist
    from horovod.tensorflow.keras.callbacks import BroadcastGlobalVariablesCallback
else:
    import smdistributed.dataparallel.tensorflow as dist
    from tensorflow.keras.callbacks import Callback
  
    class BroadcastGlobalVariablesCallback(Callback):
        def __init__(self, root_rank, *args):
            super(BroadcastGlobalVariablesCallback, self).__init__(*args)
         
            self.root_rank = root_rank
          
            self.broadcast_done = False

        def on_batch_end(self, batch, logs=None):
          
            if self.broadcast_done:
                return
                
            dist.broadcast_variables(self.model.variables, root_rank=self.root_rank)
            dist.broadcast_variables(self.model.optimizer.variables(), root_rank=self.root_rank)
            self.broadcast_done = True
        
# Create Custom Model that performs the train step using 
# DistributedGradientTape

from keras.engine import data_adapter

class CustomModel(tf.keras.Model):
    @tf.function
    def train_step(self, data):
        x, y, w = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, w)
        tape = dist.DistributedGradientTape(tape)
        self._validate_target_and_loss(y, loss)
        self.optimizer.minimize(loss, 
                                self.trainable_variables,
                                tape=tape)
        return self.compute_metrics(x, y, y_pred, w)

def get_dataset(batch_size, rank):
    def parse_sample(example_proto):
        image_feature_description = {'image': tf.io.FixedLenFeature([], tf.string),
                                     'label': tf.io.FixedLenFeature([], tf.int64)}

        features = tf.io.parse_single_example(example_proto, 
                                              image_feature_description)
        image = tf.io.decode_raw(features['image'], tf.uint8)
        image.set_shape([3 * 32 * 32])
        image = tf.reshape(image, [32, 32, 3])
        image = tf.cast(image, tf.float32)/255.
        label = tf.cast(features['label'], tf.int32)
        return image, label
    #FAILED_PRECONDITION:  /opt/ml/input/data/training/validation; Is a directory
    print("SM_CHANNEL_TRAINING", os.environ.get("SM_CHANNEL_TRAINING")+'/*')
    
    import glob
    print(glob.glob(os.environ.get("SM_CHANNEL_TRAINING")+'/*'))
    print(glob.glob(os.environ.get("SM_CHANNEL_TRAINING")+'/validation/*'))
    print(glob.glob(os.environ.get("SM_CHANNEL_TRAINING")+'/train/*'))

    aut = tf.data.experimental.AUTOTUNE
    records = tf.data.Dataset.list_files(os.environ.get("SM_CHANNEL_TRAINING")+'/train/*', shuffle=True) # CIFAR 10
    #records = tf.data.Dataset.list_files(os.environ.get("SM_CHANNEL_TRAINING")+'/*', shuffle=True) # MINIST SHARDED 120
    ds = tf.data.TFRecordDataset(records, num_parallel_reads=aut)
    ds = ds.repeat()
    ds = ds.map(parse_sample, num_parallel_calls=aut)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(aut)
    return ds

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Train resnet")
    parser.add_argument("--model_dir", type=str, 
                        default="./model_keras_resnet")

    args = parser.parse_args()

    # init distribution lib
    dist.init()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    #print("GPUs", gpus)
    
    for gpu in gpus:
        # NOT SURE WHY WE NEED TO ITERATE THROUGH ALL GPUs AS we do it multiple times as THIS CODE RUNS for each proccess (=as many times as ther eare GPUs!) so we are setting it multiple times.
        #print("gpu in gpus", dist.local_rank(), gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
        
    # this code is run once per process!!!
    # Thus it will run as many times as there are GPUS!!!
    if gpus:
        #print("gpus[dist.local_rank()", dist.local_rank(), gpus[dist.local_rank()])
        #print("LOGICAL 1", tf.config.experimental.list_logical_devices('GPU'))
        #print("PHYSICAL 1", tf.config.experimental.list_physical_devices('GPU'))
        
        # REQUIRES; this will set logical_device to 0 for every process, but the visible device will keep the dist.local_rank()!!!
        tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], 'GPU')
        
        # SETTING all GPUs as visible leads to GPU MEMORY errors...
        #tf.config.experimental.set_visible_devices(gpus, 'GPU')        
        #print("LOGICAL 2", dist.local_rank(), tf.config.experimental.list_logical_devices('GPU'))
        print("VISIBLE 2", dist.local_rank(), tf.config.experimental.get_visible_devices('GPU'))                                                   
        #print("PHYSICAL 2", dist.local_rank(), tf.config.experimental.list_physical_devices('GPU'))
        
    input_shape = (32, 32, 3)
    classes = 10
    inputs = tf.keras.Input(shape=input_shape)

    outputs = tf.keras.applications.ResNet50(weights=None, 
                                             input_shape=input_shape, 
                                             classes=classes)(inputs)
    model = CustomModel(inputs, outputs)
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                  optimizer= tf.optimizers.Adam(learning_rate=0.001*dist.size()))
    dataset = get_dataset(batch_size = 1024, rank=dist.local_rank())
    cbs = [BroadcastGlobalVariablesCallback(0)]
    
    if dist.local_rank() == 0:
        import time
        start_time = time.time()
    
    model.fit(dataset, 
              steps_per_epoch=10, # InExample: 100
              epochs=1,  # InExample: 10
              callbacks=cbs, 
              verbose=2)
    
    if dist.local_rank() == 0:
        print("EXEC TIME:" , time.time() - start_time)
