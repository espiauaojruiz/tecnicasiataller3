## Taller 3
Para este taller se plantea una aplicación chatbot que permita a los usuarios obtener recomendaciones sobre hoteles en Colombia basados en un dataset con esta inforamción y la tecnica de Agentes para mejorar el desempeño del modelo, para esta app se hace uso de tecnologías como ```gpt-4o```, ```Lance DB``` y agentes LLM.

En un navegador web se cargará la aplicación, epecíficamente el componente para poder realizar preguntas y obtenber respuestas (chatbot).

### Estructura del proyecto
```
/Proyecto
│
├── agente.py
│
├── visualizacion_datos.ipynb
│
├── requirements.txt
│
└── /data
    │
    └── hotel_reviews.csv
```

### Requisitos
* Python 3.10
* En el diractorio raíz del proyecto crear el archivo ```.env``` en el cual se deberá especificar el API key con la clave ```OPENAI_API_KEY```
* Instalación de las dependencias, desde el diractorio raíz del proyecto ejecutar el comando ```pip install -r requirements.txt```

### Uso
* Desde el diractorio raíz del proyecto ejecutar el comando ```streamlit run agente.py```
* Para la ejecución de las técnicas de visualización de datos, ejecutar el jupiter notebook ```visualizacion_datos.ipynb```
