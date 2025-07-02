#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


dashboad_html_code = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vLLM MindSpore Profiler Dashboard</title>
    <style>
        /* 基础重置 */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen;
        }

        /* 主题色定义 */
        :root {
            --primary: #2F80ED;
            --secondary: #4CAF50;
            --danger: #EF5350;
            --text: #2D2D2D;
            --bg: #F8F9FA;
        }

        /* 全局样式 */
        body {
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
            min-height: 100vh;
        }

        /* 头部样式 */
        .dashboard-header {
            text-align: center;
            padding: 2rem 0;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }

        /* 输入区域 */
        .input-group {
            display: grid;
            gap: 1.5rem;
            max-width: 800px;
            margin: 0 auto 2rem;
        }

        input[type="text"], input[type="password"] {
            width: 100%;
            padding: 1rem;
            border: 2px solid #E0E0E0;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }

        input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(47, 128, 237, 0.2);
        }

        /* 按钮组 */
        .button-group {
            display: flex;
            gap: 1.5rem;
            justify-content: center;
            margin-bottom: 2rem;
        }

        .btn {
            padding: 1rem 2rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-start {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
        }

        .btn-stop {
            background: linear-gradient(135deg, #FF5252, #FF1744);
            color: white;
        }

        /* 结果区域 */
        .result-container {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }

        /* 表格优化 */
        .profile-table {
            width: 100%;
            border-collapse: collapse;
            margin: 2rem 0;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        }

        .profile-table th {
            background: var(--primary);
            color: white;
            padding: 1.2rem;
            font-size: 1.1rem;
        }

        .profile-table td {
            padding: 1.2rem;
            border-bottom: 1px solid #ECECEC;
            transition: background 0.3s ease;
        }

        .profile-table tr:hover {
            background: #F6F8FA;
        }

        /* 响应式断点 */
        @media (max-width: 768px) {
            .input-group {
                grid-template-columns: 1fr;
            }

            .button-group {
                flex-direction: column;
                gap: 1rem;
            }
        }
    </style>
</head>
<body>
    <header class="dashboard-header">
        <h1>vLLM MindSpore Profiler Dashboard</h1>
    </header>

    <div class="input-group">
        <input 
            id="promptInput" 
            type="text" 
            placeholder="请输入推理提示文本..."
            value="hello, vllm-mindspore"
        >
        
        <input 
            id="modelInput" 
            type="text" 
            placeholder="模型路径"
            value="/home/lll/dockers/vllm-develop/workspace/scripts/Qwen2-7B"
        >
    </div>

    <div class="button-group">
        <button id="inferBtn" class="btn btn-start">
            <span>Infer Request</span>
            <svg class="icon" width="20" height="20" viewBox="0 0 24 24">
                <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"/>
            </svg>
        </button>
        
        <button id="startProfileBtn" class="btn btn-start">
            Start Profile
        </button>
        
        <button id="stopProfileBtn" class="btn btn-stop">
            Stop Profile
        </button>
    </div>

    <div class="result-container">
        <h2>Infer Results</h2>
        <div id="iuferResult" class="sanitize"></div>
    </div>

    <div class="button-group">
        <button id="refreshProfileResultBtn" class="btn">
            <span>Refresh Profile Results</span>
            <svg class="icon" width="20" height="20" viewBox="0 0 24 24">
                <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 2.92-7.99 7s3.57 7 7.99 7c2.21 0 4.2-1.08 5.66-2.42l2.2 2.2c-1.23 1.89-3.48 3.25-6.06 3.25-6.63 0-12-5.37-12-12 0-3.54 1.72-6.8 4.43-8.85l-2.2-2.2zM12 19c-3.31 0-6-2.69-6-6s2.69-6 6-6 6 2.69 6 6-2.69 6 6 6z"/>
            </svg>
        </button>
    </div>

    <div class="table-container">
        <table class="profile-table">
            <thead>
                <tr>
                    <th>No.</th>
                    <th>File</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody id="profileDatasTable">
                <!-- 动态内容 -->
            </tbody>
        </table>
    </div>
	
    <script>
		var server_ip = window.location.hostname
		var server_port = window.location.port
		var base_url = "http://" + server_ip + ":" + server_port + "/"
		
		function sendInferRequest() {
				infer_prompt_str = document.getElementById('promptInput').value;
				model_id_str = document.getElementById('modelInput').value;
				const data = {
					model: model_id_str,
					prompt: infer_prompt_str,
					max_tokens: 128,
					temperature: 0,
					top_p: 1.0,
					top_k: 1,
					repetition_penalty: 1.0
				};
				
				api_path = "v1/completions"
				dest_url = base_url + api_path
			    fetch(dest_url, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify(data)
				})
				.then(response => response.json())
				.then(json => {
					text_str = json["choices"][0]["text"]
					result_text = document.getElementById('iuferResult');
					result_text.textContent = text_str;
				})
				.catch(error => console.error('错误:', error));
		}
		
		document.getElementById('inferBtn').addEventListener('click', sendInferRequest)
	
		function sendStartProfile() {
				api_path = "start_profile"
				dest_url = base_url + api_path
			    fetch(dest_url, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify("")
				})
				.then(response => {})
				.catch(error => console.error('错误:', error));
		}
		
		document.getElementById('startProfileBtn').addEventListener('click', sendStartProfile)
		
		function sendStopProfile() {
				api_path = "stop_profile"
				dest_url = base_url + api_path
			    fetch(dest_url, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify("")
				})
				.then(response => {})
				.catch(error => console.error('错误:', error));
		}
		
		document.getElementById('stopProfileBtn').addEventListener('click', sendStopProfile)
		
		function sendGetProfileResults() {
				api_path = "get_profile_result_files"
				dest_url = base_url + api_path
			    fetch(dest_url, {
					method: 'GET',
					headers: {
						'Content-Type': 'application/json'
					}
				})
				.then( response => {
					console.log('成功:', response)
					return response.json()
				})
				.then(json => {
					profile_results = json["vllm_ms_profile_files"]
					console.log('成功:', profile_results)
					//base_url = "http://90.90.94.238:8060/get_profile_data/"
					
					const tbody = document.getElementById('profileDatasTable');
					tbody.innerHTML = ""
					profile_results.forEach((item, index) => {
					  const row = document.createElement('tr');
					  api_path = "get_profile_data/"
					  download_url = base_url + api_path + item
					  row.innerHTML = `
						<td>${index + 1}</td>
						<td>${item}</td>
						<td><a href="${download_url}">Download</a></td>
					  `;
					  tbody.appendChild(row);
					});
				})
				.catch(error => console.error('错误:', error));
		}
		
		document.getElementById('refreshProfileResultBtn').addEventListener('click', sendGetProfileResults)
    </script>
</body>
</html>
</html>
'''

def get_dashboard_html() -> str:
    return dashboad_html_code
