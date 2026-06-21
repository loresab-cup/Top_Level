запустить терминал в данной папке
создать виртуальное окружение командой и установить зависимости
linux: 
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
windows(cmd)
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

если вы уже устанавливали зависимости раньше выполните команду для перехода в вирутальное окружение
linux: 
source .venv/bin/activate
windows(cmd)
.venv\Scripts\activate.bat
затем запустите приложение командой
streamlit run app.py
