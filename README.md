
# Fairness of Content Creators with Different Posting Frequency

## Project Overview
Inspired by [IFofSMI](https://github.com/StefaniaI/ABM-IFforSMI), this project aims to simulate the process of interactions of content creators(CCs) and non-creator users(seekers) in social media platforms. In the simulations, the frequency of content posting is considered in the recommendation systems. The results of both CCs individual fairness and user satisfaction are displayed.

## Data acquirement

To collect data from the website of [YouTubeTopChannels](https://us.youtubers.me/global), execute the file of [data_prepare.py](script/data_prepare.py) to get raw data with data of all YouTube categories.
To be noticed, the data in the website changes over time, and re-acquiring may lead to inconsistent data.
The data collected is saved in [data](data), the top 1000 subscribed channel data of 16 categories and all channels.
## Installation and Setup
The required Python packages are listed in [requirements.txt](requirements.txt).

## Code structure
The files in [simu_process](simu_process) include [model.py](simu_process/model.py) which defines the class needed for CCs, users, and the simulation process, and [simulation](simu_process/simulation.py) provides methods to realize the simulation. Other needed functions are listed in [tools](tools). The dictionary of [script](script) contains the files of execution.
### Configs
The configuration files will be generated in the dictionary of [config](config). The code for generation configuration files is in [data_analysis.ipynb](script/data_analysis.ipynb). Here only a demo configuration file is uploaded.
### Execution
To quickly run a simulation with a configuration file, you could simply run the code below:
```python
import simu_process.simulation as sim
config_list = ['config/config.yaml']
simu = sim.Simulation(config_list[0])
simu.run_and_save()
```
To run simulations for all configuration file and further analysis, [data_analysis.ipynb](script/data_analysis.ipynb) provide detailed code for analysis and execution.

