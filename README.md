# hse-bigdata-team2

## Предварительная настройка
На компьютере должен быть установлен python и git

Для Ubuntu 20.04 делаем так:
```shell
$ sudo apt update && sudo apt install -y python3-pip python3-venv git
```

Если подключение к хостам выполняется через пароль, а не через ssh-ключ, то нужно еще установить пакет `sshpass`

```shell
$ sudo apt install -y sshpass
```

Скачиваем код репозитория:

```shell
$ git clone https://github.com/TiunovNN/hse-bigdata-team2.git
$ cd hse-bigdata-team2
```

### Настроить ansible

настраиваем venv

```shell
~/hse-bigdata-team2$ python3 -m venv .venv
~/hse-bigdata-team2$ source .venv/bin/activate
~/hse-bigdata-team2$ python3 -m pip install -r requirements.txt
```

Проверяем что ansible готов к работе:

```shell
$ ansible --version
ansible [core 2.17.5]
```

### Настроить ssh

Ко всем узла должен быть настроен ssh.

Например, настройках трех узлов `~/.ssh/config`
```config
Host jumpnode
    HostName 192.168.1.10
    User team

Host node1
    HostName 192.168.1.11
    # ProxyJump jumpnode  # если подключаемся через jump-ноду
    User team

Host node2
    HostName 192.168.1.12
    # ProxyJump jumpnode  # если подключаемся через jump-ноду
    User team

Host node3
    HostName 192.168.1.13
    # ProxyJump jumpnode  # если подключаемся через jump-ноду
    User team
```

### Настроить inventory/hosts.yaml

Если названия нод отличаются от тех, что указаны в [host.yaml](inventory/hosts.yaml), то там нужно подставить правильные названия

Если sudo на хостах требует ввода пароля, то нужно раскоментировать строки `#ansible_become_password: secretpassword` и заменить `secretpassword` на свой.

Если подключение к хостам выполняется через пароль, а не через ssh-ключ, то нужно раскоментировать строки `#ansible_ssh_pass: secretpassword` и заменить `secretpassword` на свой.

Например:
```yaml
datanodes:
  hosts:
    node1:
      ansible_become: yes
      ansible_user: team
      ansible_become_password: 'secretpassword'
      ansible_ssh_pass: 'secretpassword' # одинарные кавычки обязательны
```

Чтобы убедится, что правильно настроили нужно выполнить следующую команду:

```shell
$ ansible -i inventory -m ping all
```

### Изменить пароль для авторизации в nginx

Напишите свой пароль для авторизации в nginx в файл  [all.yaml](inventory/group_vars/all.yaml)

## Homework 1

### Описание

1. Необходимо развернуть кластер hdfs включающий в себя 3 DataNode. Должны быть запущены и выполняться следующие демоны: NameNode, Secondary NameNode и три DataNode
2. Кластер должен быть развернут на разных виртуальных машинах, которые будут вам предоставлены. Информация о хостах появится в чате курса.
3. Распределение демонов зависит от числа хостов. Более подробная информация будет доступна, когда станет известно количество предоставляемых виртуальных машин.
4. Кластер должен быть целостным, в логах демонов не должно быть сообщений о критических ошибках и они должны быть в работоспособном состоянии
Как понять, что кластер целостный:
Вариант 1. Зайти в интерфейс NameNode в Hadoop. В нем не должны быть деградировавших нод, должны присутстовать 3 работающих DataNode.
Вариант 2. В логах кластера не должно быть критических ошибок
5. Ограничения по операционной системе: Ubuntu 20 и Debian 10
6. На узлах должно обеспечение по ssh
7. Виртуальные машины будут чистые. Пишите инструкции так, как будто вам нужно научить кого-то разоврачивать кластер.
8. Этот кластер будет использован для выполнения последующих практических заданий

### Запуск настройки

```shell
ansible-playbook -i inventory playbooks/configure_hdfs.yml
```


## Homework 2
Развернуть YARN и опубликовать веб-интерфейсы основных и вспомогательных демонов кластера для внешнего использования.
Поднять yarn с history сервером и опубликовать webui всех датанод, непосредственно кластера и Historyjob сервера, чтобы мы могли их видеть через общий веб интерфейс.

### Запуск настройки

```shell
ansible-playbook -i inventory playbooks/configure_yarn.yml
```

## Homework 3
Развертывание Apache Hive

1. Установить hive
2. Развернуть Hive в конфигурации пригодной для производственной эксплуатации, т.е.
с отдельным хранилищем метаданных.
3. Загрузить данные на hdfs
4. Трансформировать загруженные данные в таблицу Hive.
5. Преобразовать полученную таблицу в партиционированную.


### Запуск настройки

```shell
ansible-playbook -i inventory playbooks/configure_hive.yml
```

### Работа с данными

**TODO**

## Homework 4

Использование Apache Spark под управлением YARN для чтения, трансформации и записи данных

* Запустить сессию Apache Spark под управлением YARN, развернутого кластера в предыдущих заданиях
* Подключиться к кластеру HDFS развернутому в предыдущих заданиях
* Используя созданную ранее сессию Spark прочитать данные, которые были предварительно загружены на HDFS
* Провести несколько трансформаций данных (например, агрегацию или преобразование типов)
* Сохранить данные как таблицу
* Убедиться, что стандартный клиент hive может прочитать данные.
* Для повышения оценки: применить 5 трансформаций разных видов, сохранить данные как партиционированную таблицу.


### Запуск настройки

```shell
ansible-playbook -i inventory playbooks/configure_spark.yml
```

### Работа с данными

## Homework 5

Использование реализация регулярного процесса чтения, трансформации и записи данных под управлением оркестратора (Prefect или Apache Airflow по выбору)

📌 В рамках потока (prefect) или ОАГ (Airflow)
* Запустить сессию Apache Spark под управлением YARN, развернутого кластера в предыдущих заданиях
* Подключиться к кластеру HDFS развернутому в предыдущих заданиях
* Используя созданную ранее сессию Spark, прочитать данные, которые были предварительно загружены на HDFS
* Провести несколько трансформаций данных (например, агрегацию или преобразование типов)
* Сохранить данные как таблицу


### Запуск настройки

```shell
ansible-playbook -i inventory playbooks/configure_prefect.yml
```
В данном случае процесс запускается в фоновом режиме. 
Чтоб его убить, надо найти его PID через `ps aux | grep python` и убить через `kill PID`.
Лучше способа реализовать работу флоу в фоновом режиме без контейнера не предложили даже разработчики:
https://github.com/PrefectHQ/prefect/issues/11522#issuecomment-1976843871 


## Homework 6

### Задача
1. Создать таблицу с данными в GreenPlum, соответствующую вашему имени пользователя `user*`.
2. В таблице должны лежать те же данные, что использовались на предыдущих занятиях.
3. Таблица должна успешно возвращать данные по запросу `SELECT`.

### Выполнение задания

#### Шаг 1: Вход на сервер
```bash
ssh user@91.185.85.179
```

#### Шаг 2: Если bash не видит утилиту `psql`, выполняем команду
```bash
source /usr/local/greenplum-db/greenplum_path.sh
```

#### Шаг 3: Подключаемся к базе данных
```bash
psql -d idp
```

#### Шаг 4: Подготовка данных

**4.1** Подгружаем данные из репозитория:
```bash
wget https://raw.githubusercontent.com/TiunovNN/hse-bigdata-team2/master/sample_data.csv -O sample_data.csv
```

**4.2** Переименовываем файл:
```bash
mv sample_data.csv team-2-data.csv
```

**4.3** Создаем папку на сервере:
```bash
ssh user@91.185.85.179 "mkdir -p ~/team-2-data"
```

**4.4** Отправляем файл на сервер:
```bash
scp team-2-data.csv user@91.185.85.179:~/team-2-data/
```

#### Шаг 5: Запускаем `gpfdist` в отдельной вкладке консоли
```bash
cd ~/team-2-data/
gpfdist
```

#### Шаг 6: Создаем EXTERNAL таблицу
```sql
CREATE EXTERNAL TABLE team_2_data_external (
    date TEXT,
    price_A DOUBLE PRECISION,
    price_B DOUBLE PRECISION,
    price_C DOUBLE PRECISION,
    price_D DOUBLE PRECISION,
    price_E DOUBLE PRECISION
)
LOCATION ('gpfdist://0.0.0.0:8080/team-2-data/team-2-data.csv')
FORMAT 'CSV' (DELIMITER ';' HEADER);
```

#### Шаг 7: Проверяем данные в EXTERNAL таблице
```sql
SELECT * FROM team_2_data_external;
```

#### Шаг 8: Создаем внутреннюю таблицу (internal)
```sql
CREATE TABLE team_2_data_internal AS 
SELECT * FROM team_2_data_external;
```

#### Шаг 9: Проверяем данные во внутренней таблице
```sql
SELECT * FROM team_2_data_internal;
```
