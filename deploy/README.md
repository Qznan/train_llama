<h1 align="center">vllm deploy script</h1>


### 背景
整理出vllm部署脚本，其中的api_server来源于vllm源码，主要是将openai格式调用接口和langchain的generate调用接口融合在一个文件里了。使一次部署同时支持langchain调用和openai格式调用。

### 详情如下
```
一、api_server.py说明
api_server主要是从vllm.entrypoints.api_server中修改得到，原来的代码只有接口：
    @app.post("/generate") # 用以与langchain结合
现增加了openai格式调用的接口:
    @app.get("/v1/models")
    @app.post("/v1/chat/completions")
    @app.post("/v1/completions") # 这几个接口代码原来是在vllm.entrypoints.openai.api_server中

二、运行脚本
sh 01run_vllm_model.sh #详情查看脚本说明

```
