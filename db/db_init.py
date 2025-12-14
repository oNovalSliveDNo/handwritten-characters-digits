import os
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv


def load_env_file(env_path):
    """Загружает переменные окружения из файла .env"""
    if os.path.exists(env_path):
        print(f"Загружаем переменные окружения из: {env_path}")
        load_dotenv(env_path)
        return True
    else:
        print(f"Файл .env не найден: {env_path}")
        return False


def get_db_params():
    """Получает параметры подключения к БД из переменных окружения"""
    return {
        'host': os.getenv('PG_HOST', 'localhost'),
        'port': os.getenv('PG_PORT', '5432'),
        'user': os.getenv('PG_USER', 'postgres'),
        'password': os.getenv('PG_PASSWORD', 'postgres'),
        'dbname': os.getenv('PG_DB', 'handwritten_characters-digits'),
    }


def create_database(params):
    """Создает базу данных, если она не существует"""
    # Подключаемся к PostgreSQL без указания базы данных
    conn_params = params.copy()
    conn_params['dbname'] = 'postgres'  # Подключаемся к стандартной БД
    try:
        conn = psycopg2.connect(**conn_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Проверяем, существует ли база данных
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (params['dbname'],))
        exists = cur.fetchone()

        if not exists:
            print(f"Создаем базу данных: {params['dbname']}")
            cur.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(params['dbname'])
            ))
            print(f"База данных '{params['dbname']}' успешно создана")
        else:
            print(f"База данных '{params['dbname']}' уже существует")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"Ошибка при создании базы данных: {e}")
        return False
    return True


def create_schemas_and_tables(params, classes):
    """Создает схемы train и test, и таблицы для каждого класса в обеих схемах"""
    try:
        # Подключаемся к созданной базе данных
        conn = psycopg2.connect(**params)
        conn.autocommit = False
        cur = conn.cursor()

        # 1. Создаем схемы train, test и ml, если они не существуют
        print("Создаем схемы 'train', 'test' и 'ml'...")
        cur.execute("CREATE SCHEMA IF NOT EXISTS train")
        cur.execute("CREATE SCHEMA IF NOT EXISTS test")
        cur.execute("CREATE SCHEMA IF NOT EXISTS ml")

        # Получаем список существующих схем для проверки
        cur.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name IN ('train', 'test')
            ORDER BY schema_name
        """)
        existing_schemas = [row[0] for row in cur.fetchall()]
        print(f"  Созданы схемы: {', '.join(existing_schemas)}")

        # 2. Создаем таблицы для каждого класса в обеих схемах
        schemas = ['train', 'test']

        for schema in schemas:
            print(f"\nСоздаем таблицы в схеме '{schema}':")

            for class_name in classes:
                table_name = f"class_{class_name}"

                # Проверяем, существует ли таблица
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = %s 
                        AND table_name = %s
                    )
                """, (schema, table_name))
                exists = cur.fetchone()[0]

                if not exists:
                    print(f"  Создаем таблицу: {schema}.{table_name}")

                    # Создаем таблицу с основными полями
                    cur.execute(sql.SQL("""
                        CREATE TABLE {}.{} (
                            id SERIAL PRIMARY KEY,
                            image_name VARCHAR(255) NOT NULL,
                            image_data BYTEA,
                            image_shape VARCHAR(100),
                            image_dtype VARCHAR(50),
                            original_path TEXT,
                            file_size INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """).format(
                        sql.Identifier(schema),
                        sql.Identifier(table_name)
                    ))

                    # Создаем индексы - исправленная часть
                    # Используем безопасное имя для индекса (без кавычек в середине)
                    index_name = f"idx_{schema}_{class_name}_image_name"

                    cur.execute(sql.SQL("""
                        CREATE INDEX {} ON {}.{} (image_name)
                    """).format(
                        sql.Identifier(index_name),
                        sql.Identifier(schema),
                        sql.Identifier(table_name)
                    ))

                    print(f"    Таблица {schema}.{table_name} создана успешно")
                else:
                    print(f"  Таблица {schema}.{table_name} уже существует")

        # 3. Создаем таблицу логов инференса
        print("\nСоздаем таблицу логов инференса: ml.inference_logs")

        cur.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'ml'
                AND table_name = 'inference_logs'
            )
        """)

        exists = cur.fetchone()[0]

        if not exists:
            cur.execute("""
                CREATE TABLE ml.inference_logs (
                    id SERIAL PRIMARY KEY,
                    true_label VARCHAR(10),
                    predicted_label VARCHAR(10) NOT NULL,
                    confidence FLOAT NOT NULL,
                    probabilities JSONB NOT NULL,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cur.execute("""
                CREATE INDEX idx_ml_inference_logs_created_at
                ON ml.inference_logs (created_at)
            """)

            print("  Таблица ml.inference_logs создана успешно")
        else:
            print("  Таблица ml.inference_logs уже существует")

            # Проверяем, есть ли колонка source
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'ml'
                    AND table_name = 'inference_logs'
                    AND column_name = 'source'
                )
            """)

            has_source = cur.fetchone()[0]

            if not has_source:
                print("  Добавляем колонку source в ml.inference_logs")

                cur.execute("""
                    ALTER TABLE ml.inference_logs
                    ADD COLUMN source TEXT
                """)
            else:
                print("  Колонка source уже существует")

        conn.commit()

        # Выводим статистику по созданным таблицам
        print("\n" + "=" * 60)
        print("СТАТИСТИКА СОЗДАННЫХ ТАБЛИЦ:")
        print("=" * 60)

        for schema in schemas:
            cur.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = %s
            """, (schema,))
            table_count = cur.fetchone()[0]
            print(f"\nСхема '{schema}': {table_count} таблиц")

            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s
                ORDER BY table_name
            """, (schema,))
            tables = cur.fetchall()

            # Показываем первые 10 таблиц, если их много
            if tables:
                for i, table in enumerate(tables[:10], 1):
                    print(f"  {i:2d}. {table[0]}")
                if len(tables) > 10:
                    print(f"  ... и еще {len(tables) - 10} таблиц")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"Ошибка при создании схем и таблиц: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False


def get_classes_from_dataset(project_root):
    """Получает список классов из датасета"""
    data_root = os.path.join(project_root, "data")
    train_path = os.path.join(data_root, "train")

    if os.path.exists(train_path):
        classes = sorted(os.listdir(train_path))
        return classes
    else:
        print(f"Предупреждение: Путь к датасету не найден: {train_path}")
        return []


def main():
    db_init_dir = os.getcwd()
    project_root = os.path.dirname(db_init_dir)  # Только один уровень вверх
    # Путь к файлу .env
    env_path = os.path.join(project_root, ".env")

    # Загружаем переменные окружения
    if not load_env_file(env_path):
        return

    # Получаем параметры подключения
    params = get_db_params()

    print("\n" + "=" * 50)
    print("ПАРАМЕТРЫ ПОДКЛЮЧЕНИЯ:")
    print("=" * 50)
    for key, value in params.items():
        if key != 'password':  # Не показываем пароль в логах
            print(f"  {key}: {value}")
    print("=" * 50)

    # Создаем базу данных
    print("\n1. СОЗДАНИЕ БАЗЫ ДАННЫХ")
    if not create_database(params):
        print("Не удалось создать базу данных")
        return
    print("✓ База данных готова\n")

    # Получаем список классов
    print("2. ПОЛУЧЕНИЕ СПИСКА КЛАССОВ")
    classes = get_classes_from_dataset(project_root)

    if not classes:
        print("✗ Не удалось получить список классов")
        return

    print(f"✓ Найдено классов: {len(classes)}")
    if classes:
        print(f"  Первые 10 классов: {classes[:10]}")
        if len(classes) > 10:
            print(f"  ... и еще {len(classes) - 10} классов")

    # Создаем схемы и таблицы
    print(f"\n3. СОЗДАНИЕ СХЕМ И ТАБЛИЦ")
    print("-" * 50)
    if create_schemas_and_tables(params, classes):
        print("\n" + "=" * 50)
        print("ВЫПОЛНЕНО УСПЕШНО!")
        print("=" * 50)
        print(f"Создано:")
        print(f"  • База данных: {params['dbname']}")
        print(f"  • Схемы: train, test")
        print(f"  • Таблиц в каждой схеме: {len(classes)}")
        print(f"  • Всего таблиц: {len(classes) * 2}")
        print("=" * 50)
    else:
        print("\n✗ Ошибка при создании схем и таблиц")


if __name__ == "__main__":
    main()
