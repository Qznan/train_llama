一、api_server.py说明
api_server主要是从vllm.entrypoints.api_server中修改得到，原来的代码只有接口：
    @app.post("/generate") # 用以与langchain结合
现增加了openai格式调用的接口:
    @app.get("/v1/models")
    @app.post("/v1/chat/completions")
    @app.post("/v1/completions") # 这几个接口代码原来是在vllm.entrypoints.openai.api_server中

二、运行脚本
sh 01run_vllm_model.sh #详情查看脚本说明


