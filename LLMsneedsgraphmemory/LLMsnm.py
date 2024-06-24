from transformers import AutoConfig
model_name = "shenzhi-wang/Llama3-8B-Chinese-Chat" # @param {type: "string"}这里要改！！

model_config = AutoConfig.from_pretrained(model_name)

hidden_layers = model_config.num_hidden_layers
hidden_size = model_config.hidden_size
attention_heads = model_config.num_attention_heads

print("Model: "+str(model_name))
print("Hidden Layers (L): "+str(hidden_layers))
print("Hidden Size (h): "+str(hidden_size))
print("Attention Heads (a): "+str(attention_heads)+"\n")

#Number of parameters in the model (in billions)
#模型大小即model size
nb_billion_parameters = 8.03 # @param {type:"number"}这里要改！！截图看model size位置
print("Number of parameters in the model (n): "+str(nb_billion_parameters)+"B")

#Precision of the parameters in the model
#模型精度大小即后边tensor type 是多少填多少，如果量化处理则除以4。
bitwidth_model = 16 # @param {type:"integer"}这里要改！！截图里看tensor type位置
print("Bitwidth of the model's parameters (p): "+str(bitwidth_model)+"-bit")

#Precision of the parameters in the optimizer
#优化器精度，根据使用的优化器类型选择是模型精度的多少倍
bitwidth_optimizer = 8 # @param {type:"integer"}这里要改！！模型精度的0.5、1、2倍
print("Bitwidth of the optimizer's parameters (o): "+str(bitwidth_optimizer)+"-bit")

#The maximum number of tokens in a sequence
#问或答最大的tokens是多少，一般正常说话200以内即可。大段文字
seqlen = 64 # @param {type:"integer"}这里要改！！就说个hello
print("Sequence length (s): "+str(seqlen))

#The batch size
#就是一次处理通路（简单理解）
batch_size = 4 # @param {type:"integer"}这里要改！！根据上边选择，咱就是小模型选8.
print("Batch size (b): "+str(batch_size)+"\n")

def estimate_consumption():
  #34 sbh + 5as²b
  return round((34*seqlen*batch_size*hidden_size + 5*attention_heads*seqlen*seqlen*batch_size)*2/(1024**3),2)

def estimate_optimizer_size():
  return round((2*nb_billion_parameters*bitwidth_optimizer/8*(1000**3))/(1024**3),2)

def estimate_model_size():
  return round(nb_billion_parameters*bitwidth_model/8*(1000**3)/(1024**3),2)

activation_consumption = estimate_consumption()
model_consumption = estimate_model_size()
optimizer_consumption = estimate_optimizer_size()

#print("Memory consumption of the model: "+str(model_consumption)+" GB\n")
print("模型本身需要的显存: "+str(model_consumption)+" GB\n")

print("模型优化器需要的显存: "+str(optimizer_consumption)+" GB")
print("微调部分需要的显存: "+str(activation_consumption*hidden_layers)+" GB")
print("想微调这个模型我需要多少显存: "+str(model_consumption+optimizer_consumption+activation_consumption*hidden_layers)+" GB\n")

print("使用模型进行推理需要的显存: "+str(activation_consumption)+" GB")
print("想使用此模型推理总共需要的显存: "+str(model_consumption+activation_consumption)+" GB")