# Instructions to execute

1. First, create the virtual environment and activate the environment.
```ruby
virtualenv -p python3 myenv
source myenv/bin/activate
```

2. Then, we install all the required packages.
```ruby
pip install -r requirements.txt
```

3. Register custom gym environment.
```ruby
pip install -e gym-environments/
```

4. Now we are ready to train a DQN agent. To do this, we must execute the following command. Notice that inside the *train_DQN.py* there are different hyperparameters that you can configure to set the training for different topologies, to define the size of the GNN model, etc.
现在我们准备培训一名DQN特工。为此，我们必须执行以下命令。注意，在train_DQN内。py可以配置不同的超参数来设置不同拓扑的训练，定义GNN模型的大小等
```ruby
python train_DQN.py
```

5. Now that the training process is executing, we can see the DQN agent performance evolution by parsing the log files.
现在培训过程正在执行，我们可以通过解析日志文件看到DQN代理的性能演变。
```ruby
python parse.py -d ./Logs/expsample_DQN_agentLogs.txt
```

6. Finally, we can evaluate our trained model on different topologies executing the command below. Notice that in the *evaluate_DQN.py* script you must modify the hyperparameters of the model to match the ones from the trained model.
最后，我们可以执行下面的命令，在不同的拓扑上评估经过训练的模型。注意，在evaluate_DQN中。script必须修改模型的超参数，以匹配训练模型中的超参数。
```ruby
python evaluate_DQN.py -d ./Logs/expsample_DQN_agentLogs.txt
```

python 中的egg文件；类似于Java 中的jar 包，把一系列的python源码文件、元数据文件、其他资源文件 zip 压缩，
重新命名为.egg 文件，目的是作为一个整体进行发布。
