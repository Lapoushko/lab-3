# DA-in-GameDev-lab3
# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
РАЗРАБОТКА СИСТЕМЫ МАШИННОГО ОБУЧЕНИЯ.

Отчет по лабораторной работе #3 выполнил(а):
- Довгий Вадим Игоревич
- РИ-210942
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.

## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity.
1) Добавил ML Agent в Юнити, создал объекты на сцене и начал писать команды для активации ML агента
![2022-11-10_15-50-49](https://user-images.githubusercontent.com/45125347/201104542-78678fa8-9a74-48c2-b943-f71ee5e324b9.png)

2) Написал скрипт для работы с объектом
```cs
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;

    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;

    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }

    public float forceMultiplier = 10;

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```

3) Добавил Decision Requester, Behavior Parameters к объекту сферы
![2022-11-10_16-14-18](https://user-images.githubusercontent.com/45125347/201105031-88c9c44d-29f3-4d0e-80bb-d388bddc12d9.png)

4) Добавил файл конфигурациии нейронной сети с кодом:

``` yaml

behaviors:
  RollerBall:
    trainer_type: ppo
    hyperparameters:
      batch_size: 10
      buffer_size: 100
      learning_rate: 3.0e-4
      beta: 5.0e-4
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000
```

5) Запустил работу для ML агента и проект в Unity
![2022-11-10_16-51-57](https://user-images.githubusercontent.com/45125347/201104891-0daf5d91-6133-440d-a676-59d6e88c7527.png)


6) Сделал 27 копий модели «Target Area»
![2022-11-10_17-00-53](https://user-images.githubusercontent.com/45125347/201105444-5271b395-87f7-4df4-969e-86dd1e9d78d8.png)


7) Проверил работу обученной модели

https://user-images.githubusercontent.com/45125347/201105547-96be495a-b8d7-4f7d-9a4f-4d9e3c0bda17.mp4


## Задание 2
### Подробно опишите каждую строку файла конфигурации нейронной сети. Самостоятельно найдите информацию о компонентах Decision Requester, Behavior Parameters, добавленных на сфере.

DecisionRequest - это компонент C#, который вызывает RequestDecision(процесс принятия решений) агента автоматически с заданным интервалом.
Behaviors Parameters - это компонент для настройки поведения экземпляра агента и его свойств.
```yaml
behaviors:
  RollerBall:
    trainer_type: ppo    /* Задаёт вид обученияю. Задано обучение с подкреплением */
    
    hyperparameters:
      batch_size: 10    /* Обозначает число опытов в за одну итерацию градиентного спуска. */
      buffer_size: 100    /* Размер необходимого опыта для обновления модели поведения. */
      learning_rate: 3.0e-4    /* Начальная скорость обучения */
      beta: 5.0e-4    /* Сила регуляриззации энтропии. */
      epsilon: 0.2    /* Влияет на быстроту изменения поведения во время обучения. */
      lambd: 0.99    /* lambd -  это параметр регуляризации. */
      num_epoch: 3    /* Число эпох . */
      learning_rate_schedule: linear    /* Параметр изменения скорости обучения с течением времени. */
    
    network_settings:
      normalize: false    /* Параметр, определяющий автоматическое нормализование обучения. */
      hidden_units: 128    /* Число нейронов в скрытых слоях. */
      num_layers: 2    /* Число скрытых слоёв сети */
    
    reward_signals:
      extrinsic:
        gamma: 0.99    /* Здесь задаётся коэффициент поощерения, который должен быть меньше единицы. */
        strength: 1.0    /* Коэффициент силы, на который умножается поощерение. */
    
    max_steps: 500000    /* Задаёт максимальное количество повторов симуляции сцены. */
    time_horizon: 64    /* Количество циклов ML агента, хранящихся в буфере до ввода в модель. */
    summary_freq: 10000    /* Здесь задаётся частота сохранения статистики тренировок по шагам. */
```
## Задание 3
### Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и в первом задании, случайно изменять координаты на плоскости.

1) Добавил на сцену второй куб
![image](https://user-images.githubusercontent.com/100460661/197787006-2b4ab6bd-51e6-48b2-9824-3930d8cd9836.png)

2) Модернизировал скрипт

```cs

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public Transform Target2;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
        Target2.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(Target2.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);
        float distanceToTarget2 = Vector3.Distance(this.transform.localPosition, Target2.localPosition);

        if(distanceToTarget < 1.42f || distanceToTarget2 < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```
3) Изменил Behaviour parameters
![image](https://user-images.githubusercontent.com/100460661/197793016-91fbb5be-9402-494c-9893-78bc93855659.png)


4) Запустил обучение 
![image](https://user-images.githubusercontent.com/100460661/197792340-ae293fef-174e-461c-b441-eeb29403c9d0.png)
![image](https://user-images.githubusercontent.com/100460661/197795537-9a1bfe0a-6a0f-4508-98e0-3ed2123725c7.png)

5) Проверил работоспособность модели


https://user-images.githubusercontent.com/100460661/197796914-00425888-2d12-4095-8696-e1235f4a1d57.mp4



## Выводы

В этой лабораторной работе я научился интегрировать нейронную сеть в проект unity.

Игровой баланс - это баланс между сложностью игры и удовольствием от её геймплея. Для настройки этого баланса может понадобитсья много сил и времени, однако благодаря машинному обучению задача настройки упрощается. ИИ помогает в выявлении дисбаланса, а также помогает настроить поведение NPC.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
