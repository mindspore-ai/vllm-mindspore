# HOW TO USE DASHBOARD
## 1. prepare
- download aisbench code when you need to run dataset:

~~~
git clone https://gitee.com/aisbench/benchmark.git
~~~

- change the env and path in run_dashboard.sh

## 2. run dashboard

~~~
bash run_dashboard.sh
~~~

## 3. check results
when run_dashboard.sh finished, the results will be saved in `results` folder

~~~
cd results;
python -m http.server 8001
~~~

open the web: https://localhost:8001
