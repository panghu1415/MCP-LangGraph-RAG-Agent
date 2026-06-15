from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any

import httpx

from config import settings
from rag.vector_store import TravelKnowledgeBase, get_knowledge_base


_knowledge_base: TravelKnowledgeBase | None = None
CITY_PROFILES: dict[str, dict[str, Any]] = {
    "上海": {
        "weather_query": "Shanghai,CN",
        "attractions": ["外滩", "南京路步行街", "豫园", "陆家嘴", "上海博物馆"],
        "food": ["本帮菜", "生煎", "小笼包"],
        "ticket": 260,
        "local_transport": 90,
    },
    "上海迪士尼": {
        "weather_query": "Shanghai,CN",
        "attractions": ["创极速光轮", "加勒比海盗", "飞跃地平线", "城堡烟花", "迪士尼小镇"],
        "food": ["园区主题餐厅", "迪士尼小镇餐饮"],
        "ticket": 520,
        "local_transport": 120,
    },
    "北京": {
        "weather_query": "Beijing,CN",
        "attractions": ["故宫", "景山公园", "天坛", "颐和园", "国家博物馆"],
        "food": ["北京烤鸭", "炸酱面", "铜锅涮肉"],
        "ticket": 220,
        "local_transport": 90,
    },
    "杭州": {
        "weather_query": "Hangzhou,CN",
        "attractions": ["西湖", "灵隐寺", "法喜寺", "河坊街", "京杭大运河"],
        "food": ["西湖醋鱼", "龙井虾仁", "片儿川"],
        "ticket": 180,
        "local_transport": 80,
    },
    "成都": {
        "weather_query": "Chengdu,CN",
        "attractions": ["大熊猫繁育研究基地", "武侯祠", "锦里", "宽窄巷子", "人民公园"],
        "food": ["火锅", "串串", "担担面"],
        "ticket": 200,
        "local_transport": 75,
    },
    "广州": {
        "weather_query": "Guangzhou,CN",
        "attractions": ["陈家祠", "沙面", "广州塔", "越秀公园", "珠江夜游"],
        "food": ["早茶", "肠粉", "烧腊"],
        "ticket": 220,
        "local_transport": 80,
    },
    "深圳": {
        "weather_query": "Shenzhen,CN",
        "attractions": ["深圳湾公园", "华侨城创意园", "莲花山公园", "世界之窗", "大梅沙"],
        "food": ["粤式茶点", "潮汕牛肉火锅"],
        "ticket": 260,
        "local_transport": 85,
    },
    "南京": {
        "weather_query": "Nanjing,CN",
        "attractions": ["中山陵", "明孝陵", "夫子庙", "秦淮河", "南京博物院"],
        "food": ["鸭血粉丝汤", "盐水鸭", "小笼包"],
        "ticket": 180,
        "local_transport": 75,
    },
    "苏州": {
        "weather_query": "Suzhou,CN",
        "attractions": ["拙政园", "苏州博物馆", "平江路", "虎丘", "山塘街"],
        "food": ["苏式面", "松鼠桂鱼", "桂花糖藕"],
        "ticket": 190,
        "local_transport": 70,
    },
    "西安": {
        "weather_query": "Xi'an,CN",
        "attractions": ["兵马俑", "陕西历史博物馆", "大雁塔", "钟鼓楼", "回民街"],
        "food": ["肉夹馍", "羊肉泡馍", "凉皮"],
        "ticket": 260,
        "local_transport": 80,
    },
    "重庆": {
        "weather_query": "Chongqing,CN",
        "attractions": ["洪崖洞", "解放碑", "长江索道", "磁器口", "李子坝"],
        "food": ["重庆火锅", "小面", "酸辣粉"],
        "ticket": 180,
        "local_transport": 85,
    },
}
KNOWN_CITIES = list(CITY_PROFILES)
AMAP_ADDRESS_ALIASES = {
    "上海迪士尼": "上海迪士尼度假区",
}
TRAIN_STATION_CODES = {
    "北京": "BJP",
    "北京南": "VNP",
    "上海": "SHH",
    "上海虹桥": "AOH",
    "杭州": "HZH",
    "杭州东": "HGH",
    "南京": "NJH",
    "南京南": "NKH",
    "苏州": "SZH",
    "广州": "GZQ",
    "广州南": "IZQ",
    "深圳": "SZQ",
    "成都": "CDW",
    "重庆": "CQW",
    "西安": "XAY",
}
AIRPORT_CODES = {
    "北京": "PEK",
    "上海": "SHA",
    "杭州": "HGH",
    "南京": "NKG",
    "广州": "CAN",
    "深圳": "SZX",
    "成都": "CTU",
    "重庆": "CKG",
    "西安": "XIY",
}


def _exception_reason(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _normalize_amap_address(address: str) -> str:
    return AMAP_ADDRESS_ALIASES.get(address, address)


def _base_city(place: str) -> str:
    if "迪士尼" in place:
        return "上海"
    for city in KNOWN_CITIES:
        if city != "上海迪士尼" and city in place:
            return city
    return place


def _extract_city(text: str, fallback: str = "上海") -> str:
    for city in KNOWN_CITIES:
        if city in text:
            return city
    return fallback


def _extract_origin_destination(text: str) -> tuple[str, str]:
    city_pattern = "|".join(KNOWN_CITIES)
    patterns = [
        rf"从\s*({city_pattern})\s*(?:出发)?\s*(?:去|到|前往)\s*({city_pattern})",
        rf"({city_pattern})\s*(?:去|到|前往)\s*({city_pattern})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1), match.group(2)

    mentioned = [city for city in KNOWN_CITIES if city in text]
    if len(mentioned) >= 2:
        return mentioned[0], mentioned[1]
    if len(mentioned) == 1:
        return "出发地", mentioned[0]
    return "出发地", "上海"


def _extract_days(text: str) -> int:
    digit_match = re.search(r"(\d+)\s*[天日]", text)
    if digit_match:
        return int(digit_match.group(1))

    chinese_numbers = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
    }
    chinese_match = re.search(r"([一二两三四五六七])\s*[天日]", text)
    if chinese_match:
        return chinese_numbers[chinese_match.group(1)]
    return 2 if "周末" in text else 1


def _fallback_weather(city: str, date: str | None = None, reason: str | None = None) -> dict[str, Any]:
    source = "mock fallback: missing OPENWEATHER_API_KEY" if reason == "missing_key" else "mock"
    if reason and reason != "missing_key":
        source = f"mock fallback: {reason}"
    return {
        "city": city,
        "date": date or "周末/出行日",
        "source": source,
        "weather": "多云，可能有短时阵雨",
        "temperature": "18-25 C",
        "suggestion": "建议携带轻便雨具，户外项目优先安排在上午或天气稳定时段。",
    }


async def _openweather_geocode(city: str) -> tuple[float, float] | None:
    url = "https://api.openweathermap.org/geo/1.0/direct"
    query = CITY_PROFILES.get(city, {}).get("weather_query", city)
    params = {"q": query, "limit": 1, "appid": settings.openweather_api_key}
    async with httpx.AsyncClient(timeout=settings.request_timeout, trust_env=False) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    if not payload:
        return None
    return float(payload[0]["lat"]), float(payload[0]["lon"])


async def weather_query(city: str, date: str | None = None) -> dict[str, Any]:
    if not settings.openweather_api_key and not settings.amap_api_key:
        return _fallback_weather(city, date, "missing_key")

    openweather_error: str | None = None
    try:
        if settings.openweather_api_key:
            coords = await _openweather_geocode(city)
            if coords is None:
                openweather_error = "OpenWeather geocode empty"
            else:
                lat, lon = coords
                url = "https://api.openweathermap.org/data/2.5/weather"
                params = {
                    "lat": lat,
                    "lon": lon,
                    "appid": settings.openweather_api_key,
                    "units": "metric",
                    "lang": "zh_cn",
                }
                async with httpx.AsyncClient(timeout=settings.request_timeout, trust_env=False) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    payload = response.json()

                weather = payload["weather"][0]["description"]
                temperature = payload["main"]["temp"]
                feels_like = payload["main"].get("feels_like")
                wind_speed = payload.get("wind", {}).get("speed")
                return {
                    "city": city,
                    "date": date or "当前",
                    "source": "OpenWeather",
                    "weather": weather,
                    "temperature": f"{temperature} C",
                    "feels_like": f"{feels_like} C" if feels_like is not None else None,
                    "wind_speed": wind_speed,
                    "suggestion": "根据实时天气调整衣物、雨具和户外项目顺序。",
                }
    except Exception as exc:
        openweather_error = f"{type(exc).__name__}: {exc}"

    try:
        if settings.amap_api_key:
            return await _amap_weather(city, date=date, upstream_error=openweather_error)
    except Exception as exc:
        amap_error = _exception_reason(exc)
        reason = f"OpenWeather={openweather_error}; AMapWeather={amap_error}"
        return _fallback_weather(city, date, reason)

    return _fallback_weather(city, date, openweather_error or "missing_key")


async def _amap_geocode_detail(address: str) -> dict[str, Any] | None:
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {"key": settings.amap_api_key, "address": _normalize_amap_address(address)}
    async with httpx.AsyncClient(timeout=settings.request_timeout, trust_env=False) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    geocodes = payload.get("geocodes", [])
    if not geocodes:
        return None
    return geocodes[0]


async def _amap_geocode(address: str) -> str | None:
    detail = await _amap_geocode_detail(address)
    if not detail:
        return None
    return detail.get("location")


def _haversine_km(origin_location: str, destination_location: str) -> float:
    lon1, lat1 = [float(item) for item in origin_location.split(",")]
    lon2, lat2 = [float(item) for item in destination_location.split(",")]
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return round(radius * 2 * math.asin(math.sqrt(a)), 1)


async def _amap_weather(city: str, date: str | None = None, upstream_error: str | None = None) -> dict[str, Any]:
    detail = await _amap_geocode_detail(city)
    if not detail:
        raise RuntimeError("AMap geocode empty for weather")
    adcode = detail.get("adcode")
    if not adcode:
        raise RuntimeError("AMap geocode missing adcode")

    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {"key": settings.amap_api_key, "city": adcode, "extensions": "base"}
    async with httpx.AsyncClient(timeout=settings.request_timeout, trust_env=False) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()

    lives = payload.get("lives", [])
    if not lives:
        raise RuntimeError(f"AMap weather empty: {payload}")
    live = lives[0]
    source = "AMap weather"
    if upstream_error:
        source += f" after OpenWeather fallback: {upstream_error}"
    return {
        "city": city,
        "date": date or live.get("reporttime", "当前"),
        "source": source,
        "weather": live.get("weather", "未知"),
        "temperature": f"{live.get('temperature', '未知')} C",
        "humidity": live.get("humidity"),
        "wind_direction": live.get("winddirection"),
        "wind_power": live.get("windpower"),
        "suggestion": "根据高德实时天气调整户外景点顺序，雨天优先安排室内项目。",
    }


async def amap_search_poi(
    city: str,
    keywords: str,
    types: str = "",
    limit: int = 5,
) -> dict[str, Any]:
    if not settings.amap_api_key:
        return {"source": "mock fallback: missing AMAP_API_KEY", "city": city, "keywords": keywords, "pois": []}

    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": settings.amap_api_key,
        "keywords": keywords,
        "city": city.replace("迪士尼", ""),
        "citylimit": "true",
        "offset": min(max(limit, 1), 20),
        "page": 1,
        "extensions": "base",
    }
    if types:
        params["types"] = types

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout, trust_env=False) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
        pois = []
        for poi in payload.get("pois", [])[:limit]:
            pois.append(
                {
                    "name": poi.get("name"),
                    "type": poi.get("type"),
                    "address": poi.get("address"),
                    "location": poi.get("location"),
                    "distance": poi.get("distance"),
                }
            )
        return {"source": "AMap POI", "city": city, "keywords": keywords, "types": types, "pois": pois}
    except Exception as exc:
        return {
            "source": f"mock fallback: AMap POI {_exception_reason(exc)}",
            "city": city,
            "keywords": keywords,
            "types": types,
            "pois": [],
        }


async def _amap_location(address: str) -> tuple[str | None, str]:
    try:
        detail = await _amap_geocode_detail(address)
        location = (detail or {}).get("location")
        if location:
            return location, "AMap geocode"
        geocode_reason = "AMap geocode empty"
    except Exception as exc:
        geocode_reason = f"AMap geocode {_exception_reason(exc)}"

    city = _base_city(address)
    keywords = _normalize_amap_address(address)
    poi = await amap_search_poi(city, keywords, limit=1)
    pois = poi.get("pois", [])
    if pois and pois[0].get("location"):
        return pois[0]["location"], f"AMap POI location after geocode fallback: {geocode_reason}"
    return None, f"{geocode_reason}; POI location empty"


async def amap_search_nearby_poi(
    location: str,
    keywords: str,
    types: str = "",
    radius: int = 5000,
    limit: int = 5,
) -> dict[str, Any]:
    if not settings.amap_api_key:
        return {"source": "mock fallback: missing AMAP_API_KEY", "location": location, "keywords": keywords, "pois": []}

    url = "https://restapi.amap.com/v3/place/around"
    params = {
        "key": settings.amap_api_key,
        "location": location,
        "keywords": keywords,
        "radius": radius,
        "offset": min(max(limit, 1), 20),
        "page": 1,
        "extensions": "base",
    }
    if types:
        params["types"] = types

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout, trust_env=False) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
        pois = []
        for poi in payload.get("pois", [])[:limit]:
            pois.append(
                {
                    "name": poi.get("name"),
                    "type": poi.get("type"),
                    "address": poi.get("address"),
                    "location": poi.get("location"),
                    "distance": poi.get("distance"),
                }
            )
        return {
            "source": "AMap nearby POI",
            "location": location,
            "keywords": keywords,
            "types": types,
            "radius": radius,
            "pois": pois,
        }
    except Exception as exc:
        return {
            "source": f"mock fallback: AMap nearby POI {_exception_reason(exc)}",
            "location": location,
            "keywords": keywords,
            "types": types,
            "radius": radius,
            "pois": [],
        }


async def travel_poi_search(destination: str, interests: str = "") -> dict[str, Any]:
    city = destination.replace("迪士尼", "")
    attraction_keywords = destination if "迪士尼" in destination else f"{city} 景点"
    hotel_keywords = f"{city} 酒店"
    restaurant_keywords = f"{city} 美食"
    destination_location = None
    if settings.amap_api_key:
        try:
            destination_location, _ = await _amap_location(destination)
        except Exception:
            destination_location = None

    attractions = await amap_search_poi(city, attraction_keywords, "110000|110100|110200", limit=5)
    if destination_location:
        hotels = await amap_search_nearby_poi(destination_location, "酒店", "100000|100100|100200|100300", radius=8000, limit=3)
        restaurants = await amap_search_nearby_poi(destination_location, "餐厅|美食", "050000", radius=5000, limit=5)
    else:
        hotels = await amap_search_poi(city, hotel_keywords, "100000|100100|100200|100300", limit=3)
        restaurants = await amap_search_poi(city, restaurant_keywords, "050000", limit=5)

    profile = CITY_PROFILES.get(destination) or CITY_PROFILES.get(city, {})
    if not attractions["pois"]:
        attractions["pois"] = [{"name": name, "type": "内置城市画像", "address": city, "location": None} for name in profile.get("attractions", [])[:5]]
    if not restaurants["pois"]:
        restaurants["pois"] = [{"name": name, "type": "内置美食画像", "address": city, "location": None} for name in profile.get("food", [])[:5]]

    return {
        "source": "AMap POI + profile fallback",
        "destination": destination,
        "destination_location": destination_location,
        "interests": interests,
        "attractions": attractions,
        "hotels": hotels,
        "restaurants": restaurants,
    }


def _fallback_route(origin: str, destination: str, transport: str, reason: str | None = None) -> dict[str, Any]:
    return {
        "source": "rule-based" if not reason else f"rule-based fallback: {reason}",
        "origin": origin,
        "destination": destination,
        "transport": transport,
        "duration": "约 1.5-3 小时，视具体车次和市内换乘而定",
        "distance": None,
        "route": [
            f"{origin} 出发，优先选择高铁或城际交通",
            f"抵达 {destination} 主要交通枢纽后换乘地铁/网约车",
            "预留 30-60 分钟排队、安检和换乘缓冲",
        ],
    }


async def route_plan(origin: str, destination: str, transport: str = "高铁/地铁") -> dict[str, Any]:
    if not settings.amap_api_key:
        return _fallback_route(origin, destination, transport)

    try:
        origin_location, origin_location_source = await _amap_location(origin)
        destination_location, destination_location_source = await _amap_location(destination)
        if not origin_location or not destination_location:
            return _fallback_route(
                origin,
                destination,
                transport,
                f"AMap location empty: origin={origin_location_source}; destination={destination_location_source}",
            )

        if transport in {"自驾", "驾车", "开车"}:
            url = "https://restapi.amap.com/v3/direction/driving"
            params = {
                "key": settings.amap_api_key,
                "origin": origin_location,
                "destination": destination_location,
                "extensions": "base",
            }
            async with httpx.AsyncClient(timeout=settings.request_timeout, trust_env=False) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()

            paths = payload.get("route", {}).get("paths", [])
            if not paths:
                return _fallback_route(origin, destination, transport, "AMap route empty")
            first_path = paths[0]
            duration_minutes = round(int(first_path.get("duration", 0)) / 60)
            distance_km = round(int(first_path.get("distance", 0)) / 1000, 1)
            steps = [step.get("instruction", "") for step in first_path.get("steps", [])[:8]]
            return {
                "source": "AMap driving",
                "origin": origin,
                "destination": destination,
                "transport": transport,
                "duration": f"约 {duration_minutes} 分钟",
                "distance": f"{distance_km} 公里",
                "origin_location_source": origin_location_source,
                "destination_location_source": destination_location_source,
                "route": [step for step in steps if step],
            }

        return {
            "source": "AMap location + rule-based transit",
            "origin": origin,
            "destination": destination,
            "transport": transport,
            "duration": "城际公共交通建议以高铁/城际铁路实时班次为准",
            "distance": f"{_haversine_km(origin_location, destination_location)} 公里",
            "origin_location_source": origin_location_source,
            "destination_location_source": destination_location_source,
            "route": [
                f"{origin} 坐高铁或城际铁路前往 {destination}",
                "抵达后使用地铁、公交或网约车完成市内接驳",
                f"高德已确认地点坐标：{origin_location} -> {destination_location}",
            ],
        }
    except Exception as exc:
        return _fallback_route(origin, destination, transport, _exception_reason(exc))


def _default_trip_date() -> str:
    from datetime import date, timedelta

    return (date.today() + timedelta(days=1)).isoformat()


async def search_train_12306(origin: str, destination: str, travel_date: str | None = None) -> dict[str, Any]:
    travel_date = travel_date or _default_trip_date()
    origin_city = _base_city(origin)
    destination_city = _base_city(destination)
    from_code = TRAIN_STATION_CODES.get(origin_city) or TRAIN_STATION_CODES.get(f"{origin_city}东")
    to_code = TRAIN_STATION_CODES.get(destination_city) or TRAIN_STATION_CODES.get(f"{destination_city}虹桥")

    if settings.train_api_url:
        try:
            headers = {"Authorization": f"Bearer {settings.train_api_key}"} if settings.train_api_key else {}
            params = {"from": origin_city, "to": destination_city, "date": travel_date}
            async with httpx.AsyncClient(timeout=settings.request_timeout, trust_env=False) as client:
                response = await client.get(settings.train_api_url, params=params, headers=headers)
                response.raise_for_status()
            return {"source": "custom train API", "origin": origin_city, "destination": destination_city, "date": travel_date, "trains": response.json()}
        except Exception as exc:
            return _fallback_train(origin_city, destination_city, travel_date, f"custom API {_exception_reason(exc)}")

    if from_code and to_code:
        url = "https://kyfw.12306.cn/otn/leftTicket/query"
        params = {
            "leftTicketDTO.train_date": travel_date,
            "leftTicketDTO.from_station": from_code,
            "leftTicketDTO.to_station": to_code,
            "purpose_codes": "ADULT",
        }
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout, trust_env=False, verify=False) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
            rows = payload.get("data", {}).get("result", [])
            trains = []
            for row in rows[:5]:
                parts = row.split("|")
                if len(parts) < 33:
                    continue
                trains.append(
                    {
                        "train_no": parts[3],
                        "from_station_code": parts[6],
                        "to_station_code": parts[7],
                        "start_time": parts[8],
                        "arrive_time": parts[9],
                        "duration": parts[10],
                        "business_seat": parts[32] or "--",
                        "first_class": parts[31] or "--",
                        "second_class": parts[30] or "--",
                    }
                )
            if trains:
                return {
                    "source": "12306 public leftTicket",
                    "origin": origin_city,
                    "destination": destination_city,
                    "date": travel_date,
                    "trains": trains,
                    "note": "12306 余票和票价实时变化，请以官方为准。",
                }
        except Exception as exc:
            return _fallback_train(origin_city, destination_city, travel_date, f"12306 {_exception_reason(exc)}")

    return _fallback_train(origin_city, destination_city, travel_date, "station code missing")


def _fallback_train(origin: str, destination: str, travel_date: str, reason: str) -> dict[str, Any]:
    return {
        "source": f"train fallback: {reason}",
        "origin": origin,
        "destination": destination,
        "date": travel_date,
        "trains": [
            {
                "train_no": "G/D 推荐车次待实时查询",
                "start_time": "08:00-10:00",
                "arrive_time": "约 1.5-3 小时后",
                "duration": "约 1.5-3 小时",
                "second_class": "以 12306 实时票价为准",
            }
        ],
        "note": "当前为 fallback 数据。配置 TRAIN_API_URL 或使用可访问的 12306 服务后可返回实时车次。",
    }


async def search_flight(origin: str, destination: str, travel_date: str | None = None) -> dict[str, Any]:
    travel_date = travel_date or _default_trip_date()
    origin_city = _base_city(origin)
    destination_city = _base_city(destination)

    if settings.flight_api_url:
        try:
            headers = {"Authorization": f"Bearer {settings.flight_api_key}"} if settings.flight_api_key else {}
            params = {"from": origin_city, "to": destination_city, "date": travel_date}
            async with httpx.AsyncClient(timeout=settings.request_timeout, trust_env=False) as client:
                response = await client.get(settings.flight_api_url, params=params, headers=headers)
                response.raise_for_status()
            return {"source": "custom flight API", "origin": origin_city, "destination": destination_city, "date": travel_date, "flights": response.json()}
        except Exception as exc:
            return _fallback_flight(origin_city, destination_city, travel_date, f"custom API {_exception_reason(exc)}")

    return _fallback_flight(origin_city, destination_city, travel_date, "FLIGHT_API_URL is empty")


def _fallback_flight(origin: str, destination: str, travel_date: str, reason: str) -> dict[str, Any]:
    return {
        "source": f"flight fallback: {reason}",
        "origin": origin,
        "destination": destination,
        "date": travel_date,
        "flights": [
            {
                "flight_no": f"{AIRPORT_CODES.get(origin, '---')}->{AIRPORT_CODES.get(destination, '---')}",
                "depart_time": "09:00-12:00",
                "arrive_time": "约 2-3 小时后",
                "price": "以航司/OTA 实时票价为准",
            }
        ],
        "note": "当前为 fallback 数据。配置 FLIGHT_API_URL/FLIGHT_API_KEY 后可返回实时航班。",
    }


async def smart_transport_dispatch(
    origin: str,
    destination: str,
    message: str,
    preference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preference = preference or {}
    origin_location = None
    destination_location = None
    if settings.amap_api_key:
        origin_location, _ = await _amap_location(origin)
        destination_location, _ = await _amap_location(destination)
    distance_km = _haversine_km(origin_location, destination_location) if origin_location and destination_location else None

    prefer = preference.get("transport_prefer", "AUTO")
    if any(word in message for word in ["自驾", "开车"]) or prefer == "DRIVING":
        mode = "DRIVING"
    elif any(word in message for word in ["飞机", "航班"]) or prefer == "FLIGHT":
        mode = "FLIGHT"
    elif any(word in message for word in ["高铁", "火车", "动车"]) or prefer == "TRAIN":
        mode = "TRAIN"
    elif distance_km and distance_km > 800:
        mode = "FLIGHT"
    else:
        mode = "TRAIN"

    if mode == "DRIVING":
        detail = await route_plan(origin, destination, transport="自驾")
    elif mode == "FLIGHT":
        detail = await search_flight(origin, destination)
    else:
        detail = await search_train_12306(origin, destination)

    return {
        "source": "smart_transport_dispatch",
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "distance_km": distance_km,
        "reason": f"preference={prefer}; distance_km={distance_km}",
        "detail": detail,
    }


async def hotel_price_search(destination: str, days: int, style: str, poi_data: dict[str, Any] | None = None) -> dict[str, Any]:
    city = _base_city(destination)
    if settings.hotel_api_url:
        try:
            headers = {"Authorization": f"Bearer {settings.hotel_api_key}"} if settings.hotel_api_key else {}
            params = {"city": city, "destination": destination, "days": days, "style": style}
            async with httpx.AsyncClient(timeout=settings.request_timeout, trust_env=False) as client:
                response = await client.get(settings.hotel_api_url, params=params, headers=headers)
                response.raise_for_status()
            return {"source": "custom hotel API", "destination": destination, "hotels": response.json()}
        except Exception as exc:
            source = f"hotel fallback: custom API {_exception_reason(exc)}"
        else:
            source = "custom hotel API"
    else:
        source = "hotel fallback: HOTEL_API_URL is empty"

    if poi_data is None and settings.amap_api_key:
        try:
            poi_data = await travel_poi_search(destination, "酒店")
        except Exception:
            poi_data = None

    base = {"budget": 280, "standard": 520, "comfort": 650, "business": 850, "luxury": 1500}.get(style.lower(), 520)
    poi_hotels = (poi_data or {}).get("hotels", {}).get("pois", [])
    hotels = []
    for index, hotel in enumerate(poi_hotels[:3]):
        hotels.append(
            {
                "name": hotel.get("name"),
                "address": hotel.get("address"),
                "distance": hotel.get("distance"),
                "estimated_price": base + index * 80,
            }
        )
    if not hotels:
        hotels = [{"name": f"{city}{style}酒店候选", "estimated_price": base, "address": city}]
    return {
        "source": source,
        "destination": destination,
        "style": style,
        "nights": max(days - 1, 1),
        "hotels": hotels,
        "note": "酒店价格为估算。配置 HOTEL_API_URL 后可返回实时价格。",
    }


def travel_rag_search(query: str, top_k: int = 3) -> dict[str, Any]:
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = get_knowledge_base()
    _, destination = _extract_origin_destination(query)
    if "迪士尼" in query:
        destination = "上海迪士尼"
    search_query = f"目的地:{destination}\n{query}"
    context = _knowledge_base.answer_context(search_query, top_k=top_k)
    return {"query": query, "source": _knowledge_base.backend_name, "context": context}


def itinerary_plan(
    destination: str,
    days: int,
    interests: str,
    knowledge_context: str = "",
    poi_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    days = max(1, min(days, 7))
    profile = CITY_PROFILES.get(destination) or CITY_PROFILES.get(destination.replace("迪士尼", ""), {})
    attractions = profile.get("attractions", ["核心景点", "城市地标", "本地街区", "博物馆/文化空间"])
    food = profile.get("food", ["本地美食"])
    if poi_data:
        poi_attractions = [poi["name"] for poi in poi_data.get("attractions", {}).get("pois", []) if poi.get("name")]
        poi_food = [poi["name"] for poi in poi_data.get("restaurants", {}).get("pois", []) if poi.get("name")]
        attractions = poi_attractions or attractions
        food = poi_food or food
    plan = []
    for day in range(1, days + 1):
        if day == 1:
            items = ["抵达与酒店入住", attractions[0], attractions[1] if len(attractions) > 1 else "核心景点游览", f"晚餐体验：{food[0]}"]
        elif day == days:
            items = [attractions[-2] if len(attractions) > 2 else "轻量补充景点", attractions[-1], "返程并预留交通缓冲"]
        else:
            items = [attractions[(day * 2 - 2) % len(attractions)], attractions[(day * 2 - 1) % len(attractions)], f"本地美食：{food[day % len(food)]}"]
        plan.append({"day": day, "theme": f"{destination} 第 {day} 天", "items": items})

    return {
        "destination": destination,
        "days": days,
        "interests": interests,
        "plan": plan,
        "knowledge_used": bool(knowledge_context),
    }


def budget_estimate(
    destination: str,
    days: int,
    people: int = 1,
    style: str = "standard",
    transport_data: dict[str, Any] | None = None,
    hotel_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = CITY_PROFILES.get(destination) or CITY_PROFILES.get(destination.replace("迪士尼", ""), {})
    base = {"budget": 420, "standard": 680, "comfort": 980}.get(style, 680)
    transport = 220 * people
    if transport_data and transport_data.get("mode") == "FLIGHT":
        transport = 900 * people
    elif transport_data and transport_data.get("mode") == "DRIVING":
        transport = 500 * people
    hotel = base * max(days - 1, 1)
    if hotel_data and hotel_data.get("hotels"):
        hotel = int(hotel_data["hotels"][0].get("estimated_price", base)) * max(days - 1, 1)
    food = 150 * days * people
    tickets = int(profile.get("ticket", 300)) * people
    local = int(profile.get("local_transport", 80)) * days * people
    total = transport + hotel + food + tickets + local
    return {
        "destination": destination,
        "days": days,
        "people": people,
        "style": style,
        "items": {
            "交通": transport,
            "住宿": hotel,
            "餐饮": food,
            "门票/活动": tickets,
            "市内交通": local,
        },
        "total": total,
        "note": "预算为演示估算，真实票价、酒店和门票需以实时平台为准。",
    }


def save_itinerary(content: str, filename: str | None = None) -> dict[str, Any]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = filename or f"itinerary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    safe_name = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]", "_", safe_name)
    path = settings.output_dir / safe_name
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "filename": safe_name}


def infer_trip_slots(message: str) -> dict[str, Any]:
    origin, destination = _extract_origin_destination(message)
    if "迪士尼" in message:
        destination = "上海迪士尼"
    days = _extract_days(message)
    people_match = re.search(r"(\d+)\s*(个人|人)", message)
    people = int(people_match.group(1)) if people_match else 1
    style = "comfort" if any(word in message for word in ["舒适", "高端", "亲子"]) else "standard"
    return {
        "origin": origin,
        "destination": destination,
        "days": days,
        "people": people,
        "style": style,
        "interests": message,
    }


def to_pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
