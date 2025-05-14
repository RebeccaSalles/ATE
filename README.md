# <img src="imgs/ATE-logo.png" width="12%" /> Accuracy-Time-Energy performance benchmarking (ATE)

**Rebecca Salles, Benoit Lange, Reza Akbarinia, Florent Masseglia, Esther Pacitti**

- National Institute for Research in Digital Science and Technology (INRIA), Montpellier, France
- University of Montpellier, Montpellier, France

Anomalous events are commonly observed in real-world temporal data, known as time series. Time series anomaly detection is pervasive for process monitoring in almost every scientific application. The area presents an extensive literature and several state-of-the-art methods. Currently, method selection is mainly driven by detection accuracy and runtime. However, with the rapid evolution of hardware and connected devices, massive amounts of time series data are produced at a high rate. The real-time analysis of such time series brings new demands not only for accurate and scalable solutions, but also energy consumption management. In this scenario, any improvement in energy efficiency can have a considerable impact over time both on environmental footprint and monetary expenses. However, there are currently no published works on energy efficient time series anomaly detection. 

This paper fills this gap addressing for the first time the problem of benchmarking time series anomaly detection methods based on the trade-off between accuracy, runtime, and energy consumption. We introduce a new metric for evaluating relative energy efficiency performance, called saveUp, and provide a novel methodology, inspired by skyline queries, for benchmarking methods based on a more comprehensive set of metrics, including peak power usage and total energy consumption. Experimental results based on large datasets show that our methodology is useful for selecting the methods that provide the best performances with the lowest energy impacts. Moreover, results indicate that speedup and saveUp are not always directly correlated, as believed a priori, and sometimes it is best to ``take it slow'' in favor of green applications.

# Setup

This document describes how to set up a dedicated Linux-based monitoring node to collect, store, and visualize metrics for experimental runs. The setup involves the following components:

- [InfluxDB](https://www.influxdata.com/): time-series database
- [Telegraf](https://www.influxdata.com/time-series-platform/telegraf/): metrics collection agent
- [Mosquitto (MQTT)](https://mosquitto.org/): lightweight message broker
- [Grafana](https://grafana.com/): visualization platform

---

## 🖥️ Prerequisites

- A dedicated Linux machine (tested on Debian-based distributions)
- `sudo` privileges
- Internet access to fetch packages

---

##  Installation Guide of the Monitoring Node

### InfluxDB (v1.7)

InfluxDB stores time-series metrics collected from Telegraf.

#### Installation

```bash
curl --silent --location -O https://repos.influxdata.com/influxdata-archive.key
echo "943666881a1b8d9b849b74caebf02d3465d6beb716510d86a39f6c8e8dac7515  influxdata-archive.key" \
| sha256sum --check - && cat influxdata-archive.key \
| gpg --dearmor \
| sudo tee /etc/apt/trusted.gpg.d/influxdata-archive.gpg > /dev/null

echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive.gpg] https://repos.influxdata.com/debian stable main' \
| sudo tee /etc/apt/sources.list.d/influxdata.list

sudo apt-get update && sudo apt-get install influxdb
```

### Install of grafana 
Instructions are inspired by : https://grafana.com/docs/grafana/latest/setup-grafana/installation/debian/

Grafana is a visualization tool. It is used to visualize the metrics collected by telegraf stored into the Influxdb DB. It is running on port 3000.

For this component we will use the default configuration. After the installation, you can access the Grafana web interface by navigating to `http://localhost:3000` in your web browser. The default username and password are both `admin`. The configuration throw UI will be discussed in the dedicated section.

Install the prerequisite packages:
```bash
sudo apt-get install -y apt-transport-https software-properties-common wget
```
Import the GPG key:
```bash
sudo mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
```

To add a repository for stable releases, run the following command:
```bash
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list
```

Run the following command to update the list of available packages:
```bash
# Updates the list of available packages
sudo apt-get update
```

To install Grafana OSS, run the following command:

```bash
# Installs the latest OSS release:
sudo apt-get install grafana
```

Then follow the instructions to start the service and check if it is running:
```bash
# Start the Grafana service
sudo systemctl start grafana-server
# Enable the Grafana service to start at boot
sudo systemctl enable grafana-server
# Check the status of the Grafana service
sudo systemctl status grafana-server
```

The output should be similar to this:
```bash
● grafana-server.service - Grafana instance
   Loaded: loaded (/lib/systemd/system/grafana-server.service; enabled; vendor preset: enabled)
   Active: active (running) since Wed 2023-10-04 14:00:00 CEST; 1min 30s ago
     Docs: http://docs.grafana.org
 Main PID: 1234 (grafana-server)
    Tasks: 6 (limit: 4915)
   Memory: 20.0M
   CGroup: /system.slice/grafana-server.service
           └─1234 /usr/sbin/grafana-server web
```
### Install of mqtt
Instructions are inspired by : https://mosquitto.org/download/

Mosquitto is a message broker. It is used to send messages between the different components. It is running on port 1883.

For this component we will use the default configuration. 

```bash
# Install the Mosquitto broker
 sudo apt update 
 sudo apt install -y mosquitto
 ```


Then, we need to start the mosquitto service and check if it is running:
```bash
# Start the mosquitto service
sudo systemctl status mosquitto
```

The output should be similar to this:
```bash
 ● mosquitto.service - Mosquitto MQTT v3.1/v3.1.1 Broker
      Loaded: loaded (/lib/systemd/system/mosquitto.service; enabled; vendor pr>
      Active: active (running) since Fri 2021-10-08 06:29:25 UTC; 12s ago
        Docs: man:mosquitto.conf(5)
              man:mosquitto(8)
```

This broker can be used to collect messages from the wattmeter. The IP address of the dedicated server should be specified in the MQTT configuration window of the wattmeter.


### Install of telegraf
Instructions are inspired by : https://docs.influxdata.com/telegraf/v1/install/#download-and-install-telegraf

Telegraf is a data collector agent. This agent collects system data and sends it to the Influxdb DB, because we want to monitor the energy consumption of the machine we use. We use RAPL and a dedicated wattmeter.

For this component, we will use a specific configuration file located in telegraf folder (filename: telegraf.conf).

```bash
curl --silent --location -O \
https://repos.influxdata.com/influxdata-archive.key \
&& echo "943666881a1b8d9b849b74caebf02d3465d6beb716510d86a39f6c8e8dac7515  influxdata-archive.key" \
| sha256sum -c - && cat influxdata-archive.key \
| gpg --dearmor \
| sudo tee /etc/apt/trusted.gpg.d/influxdata-archive.gpg > /dev/null \
&& echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive.gpg] https://repos.influxdata.com/debian stable main' \
| sudo tee /etc/apt/sources.list.d/influxdata.list
sudo apt-get update && sudo apt-get install telegraf

```

Then, we need to configure telegraf to use the configuration file. The configuration file is located in the telegraf folder (filename: telegraf.conf). 

```bash
# Copy the configuration file to the telegraf configuration directory
sudo cp telegraf.conf /etc/telegraf/telegraf.conf
```

Then, we need to start the telegraf service and check if it is running:

```bash
# Start the telegraf service
sudo service telegraf start
# Check the status of the telegraf service
sudo service telegraf status
```

##  Installation Guide of the test node
Copy the 'runnerTest' folder on the test node.

On the test server go to the test folder.

Install python3 dependencies: 

```bash
cd runnerTest
pip3 install -r requirements.txt
```
Update the bash script (run.sh) to feet your requirements. The bash script is used to run the test. It is located in the runnerTest folder (filename: run.sh). 



##  Setup of the dataset
Copy the dataset in a dedicated folder, for example: /data/. This test bench is compatible with dataset like Kitsune, LTDB. In this context of this paper we use a syntetic dataset set generate with GutenTAG.



# Execution

Sioply run the bash script (run.sh) to run the test.

