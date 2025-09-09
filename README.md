# video-extract

**所以代码均有ai生成!!!**
**所以代码均有ai生成!!!**
**所以代码均有ai生成!!!**

- API Server

1. /video/process
提交任务

请求方式: POST

参数|类型|必填|说明
----|----|---|---|
video_url|string|是|视频地址|

2. /task/query
任务查询

请求方式: GET

参数|类型|必填|说明
----|----|---|---|
task_id|string|是|任务ID|

- Task Worker

1. 下载视频
2. 间隔2秒对视频截图,时间可通过环境变量调整
3. 提取音频
4. 使用whisper对音频提取文字
