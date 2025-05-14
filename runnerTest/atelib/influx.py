from influxdb import InfluxDBClient
import time


def writeInit(tags,hostinflux,portinflux,logininflux,passwordinflux,dbinflux):
    json_body = [
        {
            "measurement": "analysis",
            "tags": tags,
            "time": time.time_ns(),
            "fields": {
                "value": 0
            }
        }
    ]
    client = InfluxDBClient(hostinflux, portinflux, logininflux,passwordinflux,dbinflux)
    client.create_database(dbinflux)
    client.write_points(json_body)

def writeStart( tags,hostinflux,portinflux,logininflux,passwordinflux,dbinflux):
    json_body = [
        {
            "measurement": "analysis",
            "tags": tags,
            "time": time.time_ns(),
            "fields": {
                "value": 1
            }
        }
    ]
    client = InfluxDBClient(hostinflux, portinflux, logininflux,passwordinflux,dbinflux)
    client.create_database(dbinflux)
    client.write_points(json_body)


def writeStop(tags,hostinflux,portinflux,logininflux,passwordinflux,dbinflux):
    json_body = [
        {
            "measurement": "analysis",
            "tags": tags,
            "time": time.time_ns(),
            "fields": {
                "value": 2
            }
        }
    ]
    client = InfluxDBClient(hostinflux, portinflux, logininflux,passwordinflux,dbinflux)
    client.create_database(dbinflux)
    client.write_points(json_body)