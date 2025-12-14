Запуск проекта:

1\. Скачать датасет: https://www.kaggle.com/datasets/sujaymann/handwritten-english-characters-and-digits

2\. Скачать Python 3.13.5 (Windows): https://www.python.org/ftp/python/3.13.5/python-3.13.5-amd64.exe

3\. Скачать PostgreSQL15: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads (https://sbp.enterprisedb.com/getfile.jsp?fileid=1259904)

4\. Директория проекта: `C:\\...\\Desktop\\GitHub\\handwritten-characters-digits`

5\. Создать файл `.env` и написать внутри содержимое:

```bash

\# PostgreSQL connection settings

PG\_HOST=localhost

PG\_PORT=5432

PG\_USER=postgres

PG\_PASSWORD=postgres

PG\_DB=handwritten\_characters-digits

```

6\. Перенеси папку с датасетом (п.1) в data, чтобы структура проекта была такая:

```bash

handwritten-characters-digits

├── .env

├── LICENSE

├── README.md

├── app.py

├── data

│   ├── image\_labels.csv

│   ├── test

│   │   └── ...

│   └── train

│       └── ...

├── db

│   ├── \_\_init\_\_.py

│   ├── db.py

│   ├── db\_fill.py

│   └── db\_init.py

├── notebooks

│   ├── best\_model.pt

│   ├── handwritten\_cnn\_best.pt

│   └── train model.ipynb

├── requirements-lock.txt

├── requirements.txt

└── venv

```

6\. Открыть терминал в `C:\\...\\Desktop\\GitHub\\handwritten-characters-digits` и прописать:

```bash

python -m venv venv

.\\venv\\Scripts\\activate

pip install -Ur .\\requirements.txt

```

8\. Перейди в директорию `db` и запусти файлы создания БД:

```bash

cd db

python db\_init.py

python db\_fill.py

```

9\. Вернуться обратно в директорию `handwritten-characters-digits` и запустить web-приложение:

```bash

cd ..

streamlit run .\\app.py

```



