# Models

This folder contains everything related to the training of models. The `Trainer` class 
will acquire the parameters from the `config.json` and pass to `TrainFactory` class 
for it to create the specified train object. The `Trainer` will then fit this object.
