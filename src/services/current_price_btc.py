import json
import requests
from datetime import datetime

# Función para obtener el precio actual desde un archivo JSON
def get_current_price(api_key: str = "TU_API_KEY") -> float: #funciona con esa apikey
    """
    Consulta la API de Alpha Vantage para obtener el precio actual del Bitcoin en euros (BTC/EUR).

    Parámetros:
        api_key (str): Tu clave de API de Alpha Vantage.

    Retorna:
        float: Precio actual del Bitcoin en euros, o None si ocurre un error.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": "BTC",
        "to_currency": "EUR",
        "apikey": api_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extrae el precio del JSON devuelto por la API
        rate_info = data.get("Realtime Currency Exchange Rate", {})

        if rate_info:
            return rate_info
        else:
            return None

    except Exception as e:
        print(f"Error al consultar el precio: {e}")
        return None


def parse_current_price() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    price_file = f"data/prices/daily_price_{today}.json"
    
    # consultamos el precio actual y lo guardamos en un archivo JSON
    try:
        price_info = get_current_price()
        print("Información del precio obtenida:", price_info)
        if price_info:
            with open(price_file, "w", encoding="utf-8") as f:
                json.dump(price_info, f, ensure_ascii=False, indent=4)
    except Exception:
        return "Hubo un problema al obtener los datos del precio actual."

    try:
        if price_info.get('5. Exchange Rate') is None:
            return "No se pudo obtener el precio actual del Bitcoin."
        else:
            # Adaptar a la estructura de 'Realtime Currency Exchange Rate' usando solo los valores presentes en el JSON
            texto_resumen = (
                f"Análisis del mercado de {price_info.get('2. From_Currency Name', 'Bitcoin')} ({price_info.get('1. From_Currency Code', 'BTC')})\n\n"
                f"A la fecha y hora del último registro ({price_info.get('6. Last Refreshed', 'No disponible')} {price_info.get('7. Time Zone', 'No disponible')}), "
                f"el precio actual es de {price_info.get('5. Exchange Rate', 'No disponible')} EUR.\n"
                f"El precio de compra (Bid) es de {price_info.get('8. Bid Price', 'No disponible')} EUR y el precio de venta (Ask) es de {price_info.get('9. Ask Price', 'No disponible')} EUR.\n\n"
                f"Fuente de datos: Alpha Vantage.")
            return texto_resumen
    except json.JSONDecodeError:
        return "Hubo un problema al leer los datos del precio actual."

