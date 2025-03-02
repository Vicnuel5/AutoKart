# AutoKart

Este repositorio contiene un [proyecto de Unity](AAKart/) que implementa un kart autónomo. El pipeline de funcionamiento es el siguiente:

- **Realizar varias grabaciones del recorrido**, que se guardarán en la carpeta [`records`](records/).
- **Ejecutar el [cuaderno de Jupyter]** para entrenar el modelo con las grabaciones. Este cuaderno permite entrenar el modelo tanto con *Scikit-learn* como con un perceptrón multicapa programado desde cero.
- **Probar el modelo en Unity**. Una vez entrenado uno o varios modelos, se debe regresar al proyecto de Unity, activar la conducción autónoma y asignar al kart el modelo deseado. Existe varios modelos pregrabados en la [esta carpeta](AutoKart/AAKart/Assets/MLModels).
