import os
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import cv2
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm


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


def connect_to_db(params):
    """Создает подключение к базе данных"""
    try:
        conn = psycopg2.connect(**params)
        return conn
    except Exception as e:
        print(f"Ошибка подключения к БД: {e}")
        return None


def get_classes_from_dataset(data_root):
    """Получает список классов из датасета"""
    train_path = os.path.join(data_root, "train")

    if os.path.exists(train_path):
        classes = sorted(os.listdir(train_path))
        # Фильтруем только директории
        classes = [c for c in classes if os.path.isdir(os.path.join(train_path, c))]
        return classes
    else:
        print(f"Предупреждение: Путь к train датасету не найден: {train_path}")
        return []


def image_to_bytea(image_path):
    """Конвертирует изображение в байтовый массив для хранения в BYTEA"""
    try:
        # Читаем изображение
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Не удалось прочитать изображение: {image_path}")
            return None, None, None

        # Если изображение в градациях серого, добавляем измерение канала
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        # Конвертируем в байты
        success, encoded_image = cv2.imencode('.png', image)
        if not success:
            print(f"Не удалось закодировать изображение: {image_path}")
            return None, None, None

        image_bytes = encoded_image.tobytes()

        # Получаем информацию об изображении
        shape_str = str(image.shape)
        dtype_str = str(image.dtype)

        return image_bytes, shape_str, dtype_str
    except Exception as e:
        print(f"Ошибка при обработке изображения {image_path}: {e}")
        return None, None, None


def get_image_statistics(data_root, schema_name):
    """Получает статистику по изображениям для указанной схемы"""
    schema_path = os.path.join(data_root, schema_name)

    if not os.path.exists(schema_path):
        return [], 0

    classes = sorted(os.listdir(schema_path))
    classes = [c for c in classes if os.path.isdir(os.path.join(schema_path, c))]

    total_images = 0
    images_info = []

    for class_name in classes:
        class_path = os.path.join(schema_path, class_name)
        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        image_count = len(image_files)
        total_images += image_count

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            images_info.append({
                'schema': schema_name,
                'class_name': class_name,
                'image_name': img_file,
                'image_path': img_path,
                'file_size': os.path.getsize(img_path) if os.path.exists(img_path) else 0
            })

    return images_info, total_images


def insert_image_to_db(conn, schema_name, class_name, image_info):
    """Вставляет изображение в соответствующую таблицу базы данных"""
    try:
        cur = conn.cursor()

        # Подготавливаем имя таблицы
        table_name = f"class_{class_name}"

        # Конвертируем изображение в BYTEA
        image_bytes, shape_str, dtype_str = image_to_bytea(image_info['image_path'])

        if image_bytes is None:
            return False

        # Проверяем, существует ли уже это изображение
        cur.execute(sql.SQL("""
            SELECT id FROM {}.{} 
            WHERE image_name = %s
        """).format(
            sql.Identifier(schema_name),
            sql.Identifier(table_name)
        ), (image_info['image_name'],))

        existing = cur.fetchone()

        if existing:
            # Обновляем существующую запись
            cur.execute(sql.SQL("""
                UPDATE {}.{} 
                SET image_data = %s, 
                    image_shape = %s, 
                    image_dtype = %s,
                    file_size = %s,
                    original_path = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE image_name = %s
            """).format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name)
            ), (
                psycopg2.Binary(image_bytes),
                shape_str,
                dtype_str,
                image_info['file_size'],
                image_info['image_path'],
                image_info['image_name']
            ))
            operation = "обновлено"
        else:
            # Вставляем новую запись
            cur.execute(sql.SQL("""
                INSERT INTO {}.{} 
                (image_name, image_data, image_shape, image_dtype, file_size, original_path)
                VALUES (%s, %s, %s, %s, %s, %s)
            """).format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name)
            ), (
                image_info['image_name'],
                psycopg2.Binary(image_bytes),
                shape_str,
                dtype_str,
                image_info['file_size'],
                image_info['image_path']
            ))
            operation = "добавлено"

        conn.commit()
        cur.close()
        return True, operation

    except Exception as e:
        print(f"Ошибка при вставке изображения {image_info['image_name']} в {schema_name}.class_{class_name}: {e}")
        conn.rollback()
        return False, "ошибка"


def check_tables_exist(conn, schema_name, classes):
    """Проверяет существование всех необходимых таблиц"""
    try:
        cur = conn.cursor()
        missing_tables = []

        for class_name in classes:
            table_name = f"class_{class_name}"
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = %s 
                    AND table_name = %s
                )
            """, (schema_name, table_name))

            exists = cur.fetchone()[0]
            if not exists:
                missing_tables.append(table_name)

        cur.close()
        return missing_tables

    except Exception as e:
        print(f"Ошибка при проверке таблиц: {e}")
        return []


def fill_schema(conn, data_root, schema_name, classes, batch_size=100, skip_existing=False):
    """Заполняет указанную схему изображениями"""
    print(f"\n{'=' * 60}")
    print(f"ЗАПОЛНЕНИЕ СХЕМЫ: {schema_name.upper()}")
    print(f"{'=' * 60}")

    # Получаем статистику по изображениям
    print("Получаем список изображений...")
    images_info, total_images = get_image_statistics(data_root, schema_name)

    if total_images == 0:
        print(f"В схеме '{schema_name}' нет изображений")
        return 0, 0

    print(f"Найдено изображений: {total_images}")
    print(f"Найдено классов: {len(classes)}")

    # Проверяем существование таблиц
    print("\nПроверяем существование таблиц...")
    missing_tables = check_tables_exist(conn, schema_name, classes)

    if missing_tables:
        print(f"Предупреждение: Не найдены таблицы для классов:")
        for table in missing_tables[:10]:
            print(f"  - {table}")
        if len(missing_tables) > 10:
            print(f"  ... и еще {len(missing_tables) - 10} таблиц")

        response = input("\nПродолжить только с существующими таблицами? (y/n): ")
        if response.lower() != 'y':
            print("Заполнение отменено")
            return 0, 0

        # Фильтруем классы только с существующими таблицами
        existing_classes = [c for c in classes if f"class_{c}" not in missing_tables]
        print(f"Будет заполнено таблиц: {len(existing_classes)}")
    else:
        existing_classes = classes
        print("Все таблицы существуют ✓")

    # Заполняем таблицы
    print(f"\nНачинаем заполнение схемы '{schema_name}'...")

    success_count = 0
    error_count = 0
    skipped_count = 0

    # Группируем изображения по классам для удобства
    images_by_class = {}
    for img_info in images_info:
        if img_info['class_name'] in existing_classes:
            if img_info['class_name'] not in images_by_class:
                images_by_class[img_info['class_name']] = []
            images_by_class[img_info['class_name']].append(img_info)

    # Заполняем каждый класс
    for class_name in tqdm(existing_classes, desc=f"Классы схемы {schema_name}"):
        if class_name not in images_by_class:
            print(f"  Предупреждение: Для класса '{class_name}' нет изображений")
            continue

        class_images = images_by_class[class_name]
        print(f"\n  Класс '{class_name}': {len(class_images)} изображений")

        # Проверяем существующие изображения, если нужно пропускать
        existing_images = set()
        if skip_existing:
            try:
                cur = conn.cursor()
                table_name = f"class_{class_name}"
                cur.execute(sql.SQL("""
                    SELECT image_name FROM {}.{}
                """).format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name)
                ))
                existing_images = {row[0] for row in cur.fetchall()}
                cur.close()
            except Exception as e:
                print(f"    Ошибка при проверке существующих изображений: {e}")

        # Обрабатываем изображения класса
        class_success = 0
        class_errors = 0
        class_skipped = 0

        for img_info in tqdm(class_images, desc=f"  Изображения", leave=False):
            # Пропускаем существующие, если нужно
            if skip_existing and img_info['image_name'] in existing_images:
                class_skipped += 1
                continue

            # Вставляем изображение
            success, operation = insert_image_to_db(conn, schema_name, class_name, img_info)

            if success:
                class_success += 1
            else:
                class_errors += 1

        print(f"    Успешно: {class_success}, Ошибки: {class_errors}, Пропущено: {class_skipped}")
        success_count += class_success
        error_count += class_errors
        skipped_count += class_skipped

    return success_count, error_count, skipped_count


def get_database_stats(conn, schema_name):
    """Получает статистику по базе данных"""
    try:
        cur = conn.cursor()

        # Получаем общее количество записей в схеме
        cur.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables t
            CROSS JOIN LATERAL (
                SELECT COUNT(*) as row_count
                FROM information_schema.columns c
                WHERE c.table_schema = t.table_schema 
                AND c.table_name = t.table_name
                AND c.column_name = 'id'
            ) col_check
            CROSS JOIN LATERAL (
                SELECT COUNT(*) as actual_count
                FROM pg_catalog.pg_stat_user_tables s
                WHERE s.schemaname = t.table_schema 
                AND s.relname = t.table_name
            ) stat
            WHERE t.table_schema = %s
        """, (schema_name,))

        total_records = cur.fetchone()[0] or 0

        # Получаем количество таблиц
        cur.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = %s
        """, (schema_name,))

        table_count = cur.fetchone()[0]

        cur.close()
        return table_count, total_records

    except Exception as e:
        print(f"Ошибка при получении статистики БД: {e}")
        return 0, 0


def main():
    parser = argparse.ArgumentParser(description='Заполнение базы данных изображениями рукописных символов')
    parser.add_argument('--schema', choices=['train', 'test', 'both'], default='both',
                        help='Какую схему заполнять (train, test или both)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Пропускать существующие изображения')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Размер батча для обработки (по умолчанию: 100)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Путь к директории с данными (по умолчанию: ./data)')

    args = parser.parse_args()

    # Определяем корневую директорию проекта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Путь к файлу .env
    env_path = os.path.join(project_root, ".env")

    # Загружаем переменные окружения
    if not load_env_file(env_path):
        print("Не удалось загрузить переменные окружения")
        return

    # Получаем параметры подключения
    params = get_db_params()

    print("\n" + "=" * 60)
    print("ПАРАМЕТРЫ ЗАПОЛНЕНИЯ БАЗЫ ДАННЫХ")
    print("=" * 60)
    print(f"Схема: {args.schema}")
    print(f"Пропускать существующие: {'Да' if args.skip_existing else 'Нет'}")
    print(f"Размер батча: {args.batch_size}")
    print("=" * 60)

    # Определяем путь к данным
    if args.data_dir:
        data_root = args.data_dir
    else:
        data_root = os.path.join(project_root, "data")

    print(f"\nПуть к данным: {data_root}")

    if not os.path.exists(data_root):
        print(f"Ошибка: Директория с данными не найдена: {data_root}")
        return

    # Подключаемся к базе данных
    print("\nПодключаемся к базе данных...")
    conn = connect_to_db(params)

    if not conn:
        print("Не удалось подключиться к базе данных")
        return

    try:
        # Получаем список классов
        print("Получаем список классов из датасета...")
        classes = get_classes_from_dataset(data_root)

        if not classes:
            print("Ошибка: Не найдены классы в датасете")
            return

        print(f"Найдено классов: {len(classes)}")
        if classes:
            print(f"Первые 10 классов: {classes[:10]}")
            if len(classes) > 10:
                print(f"... и еще {len(classes) - 10} классов")

        # Определяем какие схемы заполнять
        schemas_to_fill = []
        if args.schema == 'both':
            schemas_to_fill = ['train', 'test']
        else:
            schemas_to_fill = [args.schema]

        # Заполняем схемы
        total_stats = {'success': 0, 'errors': 0, 'skipped': 0}

        for schema_name in schemas_to_fill:
            # Проверяем существование директории схемы
            schema_path = os.path.join(data_root, schema_name)
            if not os.path.exists(schema_path):
                print(f"\nПредупреждение: Директория схемы '{schema_name}' не найдена: {schema_path}")
                continue

            # Заполняем схему
            success, errors, skipped = fill_schema(
                conn, data_root, schema_name, classes,
                args.batch_size, args.skip_existing
            )

            total_stats['success'] += success
            total_stats['errors'] += errors
            total_stats['skipped'] += skipped

        # Выводим общую статистику
        print(f"\n{'=' * 60}")
        print("ОБЩАЯ СТАТИСТИКА ЗАПОЛНЕНИЯ")
        print(f"{'=' * 60}")
        print(f"Успешно добавлено/обновлено: {total_stats['success']}")
        print(f"Ошибок: {total_stats['errors']}")
        print(f"Пропущено: {total_stats['skipped']}")
        print(f"Всего обработано: {total_stats['success'] + total_stats['errors'] + total_stats['skipped']}")

        # Показываем текущую статистику базы данных
        print(f"\n{'=' * 60}")
        print("ТЕКУЩАЯ СТАТИСТИКА БАЗЫ ДАННЫХ")
        print(f"{'=' * 60}")

        for schema_name in ['train', 'test']:
            table_count, record_count = get_database_stats(conn, schema_name)
            print(f"\nСхема '{schema_name}':")
            print(f"  Таблиц: {table_count}")
            print(f"  Всего записей: {record_count}")

        print(f"\n{'=' * 60}")
        print("ВЫПОЛНЕНО УСПЕШНО!")
        print(f"{'=' * 60}")

    finally:
        # Закрываем соединение
        conn.close()
        print("\nСоединение с базой данных закрыто")


if __name__ == "__main__":
    main()

