# RSV22_Pochta

В конфиге задаются параметры:

* `TRAIN_DATAPATH` - путь до train датасета

* `TEST_DATAPATH` - путь до test датасета

* `cat_features` - список категориальных фичей

* `MODEL_PATH` - папка, куда сохранять или где уже сохранена модель

* `fit_model` - для обучения модели указать True, для использования обученной модели из MODEL_PATH - указать False

* `ss_path` - путь до sample_solution

* `drop_cols` - колонки для удаления

Пайплайн модели:

1) Простой препроцессинг

2) Если в категориальной фиче для какого-то значения таргет всегда 0, то исключаем из обучения, а втесте присваиваем прогноз без модели

3) Обучаем LightAutoml

4) Ставим порог отнесения к классу 1 равным 0.1

Обученная модель - https://www.dropbox.com/s/op1zhiuj89mntrr/lama_model?dl=0

Сабмит файл - https://www.dropbox.com/s/4eedgiw2jxs7lp1/lama_407860_0.1_upd.csv?dl=0

Ноутбук с воспроизведением решения - https://colab.research.google.com/drive/1MNCkD78h9GMb6-IIKQE6uA-X2CZPPg7R?usp=sharing
