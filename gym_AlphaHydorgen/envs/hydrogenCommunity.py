import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pandas as pd
import numpy as np

from model import *

'''
In this version:
1. The time step is fixed at 1 hour
2. The tank volume of H2 station is unlimited
3. Generating H2 using electricity is not supported
'''
class hydrogenCommunity(gym.Env):
    """ AlphaHydorgen is a custom Gym Environment to simulate a community equiped with on-site renewables,
    hydrogen station, hydrogen vehicles and smart grid
    ------------------------------------------------------------------------------------------------------
    Args:
        - stepLenth: length of time step, unit: s
        - building_list: a list of buildings, each element in the list is a tuple of (buildingLoad.csv, number of buildings)
            example: [('inputs/building1.csv', 10), ('inputs/building2.csv', 10), ('inputs/building3.csv', 10)]
        - pv_list: a list of on-site pvs, each element in the list is a tuple of (pvGeneration.csv, number of PVs)
            example: [('inputs/pv1.csv', 10), ('inputs/pv2.csv', 10), ('inputs/pv3.csv', 10)]
        - vehicle_list: a list of hydrogen vehicles, 
            first element is parkSchedule.csv, 
            the remaining elements in the list are tuples of 
                (vehicleParameter.csv, fuelCellEff, fuelCellCap, number of vehicles)
            example: ['inputs/vehicle_atHomeSchd.csv', ('inputs/vehicle1.csv', 100, 300, 10), 
                      ('inputs/vehicle2.csv', 100, 300, 10), ('inputs/vehicle3.csv', 100, 300, 10)]
        - station_info: a hydrogen station parameter file.
    ------------------------------------------------------------------------------------------------------
    States:
        - buildingLoad: total building load of the community, [kW]
        - pvGeneration: total on-site PV generation, [kW]
        - tank_h2Vol: total onsite-produced hydrogen stored in the station tank, [kg]
        - tank_spareVol: rest tank space for storing onsite-produced hydrogen, [kg] 
        - vehicle_park: binary variable, whether the vechile is parked at home or not
        - vehicle_max_dist: predicted maximum travel distance of today, dist_mu_wd+5*dist_sigma_wd [km]
        - vehicle_tank: hydorgen stored in the vehicle's tank, [g]
    ------------------------------------------------------------------------------------------------------
    Actions:
        - station_control: value, meaning the charge/discharge power of the hydrogen station for balancing the microgrid energy, [kW]
            positive means charging the hydrogen station for H2 production, negative means discharging the hydrogen station for powering microgrid. 
        - vehicle_charge: array, each element is H2 charge/discharge rate of each vehicle,
            positive means charge from H2 station, negative means discharge to grid/building, [g]
    """

    def __init__(self, building_list, pv_list, vehicle_list, station_info):
        '''
        In this version: 
            -- The step length is fixed at 1 hour
            
        '''
        super().__init__()
        self.episode_idx = 0
        self.time_step_idx = 0

        self.stepLenth =3600          # To be revised when the step length is not 1 hour
        self.simulationYear = 2019    # Fixed in this version
        start_time = datetime(year = self.simulationYear, month = 1, day =1)
        self.n_steps = 8760*3600//self.stepLenth   # Simulate a whole year
        freq = '{}H'.format(self.stepLenth/3600)
        self.timeIndex = pd.date_range(start_time, periods=self.n_steps, freq=freq)

        # Calculate the load for each time step
        self.buildingLoad = self._calculateBuildingLoad(building_list, self.stepLenth, self.simulationYear)
        self.pvGeneration = self._calculatePVGeneration(pv_list, self.stepLenth, self.simulationYear)

        # Initialize the hydrogen station
        self.station = Station(station_info, self.stepLenth/3600) 

        # Initialize the vehicles
        self.vehicles = []
        self.vehicle_schl_file = vehicle_list[0]
        for vehicle_tuple in vehicle_list[1:]:
            fuelCell = FuelCell(vehicle_tuple[1], vehicle_tuple[2])
            vehicle = Vehicle(vehicle_tuple[0], self.vehicle_schl_file, fuelCell, self.stepLenth)
            for _ in range(vehicle_tuple[3]):
                self.vehicles.append(vehicle)

        # define the state and action space
        vehicle_n = len(self.vehicles)           # Only control the vehicles
        self.action_names = ['station_tank'] + \
            ['vehicle_{}'.format(vehicle_i) for vehicle_i in range(vehicle_n)]
        self.actions_low = np.array([-10000] + [-100 for _ in range(vehicle_n)])    # Maximum discharging rate -100g/s
        self.actions_high = np.array([10000] + [100 for _ in range(vehicle_n)])    # Maximum charging rate 100g/s
        self.action_space = spaces.Box(low=self.actions_low,
                                       high=self.actions_high,
                                       dtype=np.float32)  

        self.obs_names = ['buildingLoad', 'pvGeneration', 'tank_h2Vol','tank_spareVol'] + \
            ['vehicle_park_{}'.format(vehicle_i) for vehicle_i in range(vehicle_n)] + \
            ['vehicle_max_dist_{}'.format(vehicle_i) for vehicle_i in range(vehicle_n)] + \
            ['vehicle_tank_{}'.format(vehicle_i) for vehicle_i in range(vehicle_n)]# + \
            #['h2Production', 'h2forGrid', 'h2forVehicle']

        self.obs_low  = np.array([0,  0,  0, 0] + [0 for _ in range(vehicle_n)] + \
            [0 for _ in range(vehicle_n)] + [0 for _ in range(vehicle_n)] + \
            [0,  0,  0])
            
        self.obs_high = np.array([10000, 10000, 10000, 10000] + [1 for _ in range(vehicle_n)] + \
            [1000 for _ in range(vehicle_n)] + [10000 for _ in range(vehicle_n)] + \
            [10000, 10000, 10000])
        self.observation_space = spaces.Box(low=self.obs_low, 
                                            high=self.obs_high, 
                                            dtype=np.float32)

    def reset(self):
        self.episode_idx += 1
        self.time_step_idx = 0
        load = self._getLoad(self.time_step_idx)
        stationTank = [0]
        stationTankVol = [0]
        stationTankSpare = [0]
        vehicles_park = []
        vehicles_max_dist = []
        vehicles_tank = []
        
        for vehicle in self.vehicles:
            vehicle_park, vehicle_max_dist, _ = self._getVihicleStateStatic(vehicle)
            vehicles_park.append(vehicle_park)
            vehicles_max_dist.append(vehicle_max_dist)           
            vehicles_tank.append(vehicle.tankVol)   # Half the tank at the begining
        obs = load + stationTankVol + stationTankSpare + vehicles_park + vehicles_max_dist + vehicles_tank# + h2Production + h2forGrid + h2forVehicle

        return obs

    def step(self, actions):
        load = self._getLoad(self.time_step_idx)
        stationTankVol = [max(self.station.tankVol, 0)]
        stationTankSpare = [self.station.capacityMax - max(self.station.tankVol, 0)]

        vehicles_park = []
        vehicles_max_dist = []
        vehicles_tank = []
        totalH2Charging = 0

        # Charge the tank of the H2 station
        if actions[0] >= 0: 
            power_H2Production, h2Production = self.station.h2Production(actions[0])
            powertoGrid = 0
            h2forGrid = 0
        elif actions[0] < 0: 
            powertoGrid, h2forGrid = self.station.powerGrid(-actions[0])
            power_H2Production = 0
            h2Production = 0
        
        for action, vehicle in zip(actions[1:], self.vehicles):
            vehicle_park, vehicle_max_dist, cruiseBackHour = self._getVihicleStateStatic(vehicle)
            if action > 0:   # Charge the vehicle tank from the H2 station
                if totalH2Charging < (self.station.chargeCap*1000):
                    realH2ChargeRate = vehicle.h2FromStation(action)
                    totalH2Charging += realH2ChargeRate
                else:
                    realH2ChargeRate = vehicle.h2FromStation(0)
            elif action < 0: # discharge the grid
                realDischargePower = vehicle.eleToGrid(-action)
                totalGridLoad -= realDischargePower
            # Vehicle's gas tank is reduced at the hour when vehicle is back 
            if cruiseBackHour:
                workingDay = self.timeIndex[self.time_step_idx].weekday()
                vehicle.cruise(workingDay)
            vehicles_park.append(vehicle_park)
            vehicles_max_dist.append(vehicle_max_dist)
            vehicles_tank.append(vehicle.tankVol)
        
        h2forVehicle = self.station.h2toVehicle(totalH2Charging)  
        h2Change = h2Production - h2forGrid - h2forVehicle
        h2netuse = -self.station.tankVol

        totalGridLoad = load[0] - 0.95*load[1] + power_H2Production - powertoGrid
        
        obs = load + stationTankVol + stationTankSpare + vehicles_park + vehicles_max_dist + vehicles_tank

        reward = (totalGridLoad, h2Change)
        done = self.time_step_idx == len(self.timeIndex)-1
        comments = (h2netuse, h2Production, h2forGrid, h2forVehicle)

        self.time_step_idx += 1
        if done:
            load = self._getLoad(self.time_step_idx-1)
        else:
            load = self._getLoad(self.time_step_idx)
        obs = load + stationTankVol + stationTankSpare + vehicles_park + vehicles_max_dist + vehicles_tank
        return obs, reward, done, comments

    

    def _calculateBuildingLoad(self, building_list, stepLenth, simulationYear):
        '''Calculate the total building load from the building list
        '''
        
        buildings = pd.DataFrame()
        for building_tuple in building_list:
            building_csv = building_tuple[0]
            building_numbers = building_tuple[1]
            building_obj = Building(building_csv, stepLenth, simulationYear)
            building = building_obj.getLoadFullYear()*building_numbers
            buildings = pd.concat([buildings,building], axis=1)
        totalLoad = buildings.sum(axis=1).values
        return totalLoad

    def _calculatePVGeneration(self, pv_list, stepLenth, simulationYear):
        '''Calculate the total PV generation from the PV list
        '''
        pvs = pd.DataFrame()
        for pv_tuple in pv_list:
            pv_csv = pv_tuple[0]
            pv_numbers = pv_tuple[1]
            pv_obj = PV(pv_csv, stepLenth, simulationYear)
            pv = pv_obj.getPowerFullYear()*pv_numbers
            pvs = pd.concat([pvs,pv], axis=1)
        totalGeneration = pvs.sum(axis=1).values
        return totalGeneration
    
    def _getLoad(self, time_step_idx):
        '''Get the building load and pv generation for the given time step
        Return a list
        '''
        load = [self.buildingLoad[time_step_idx], self.pvGeneration[time_step_idx]]
        return load
    
    def _getVihicleStateStatic(self, vehicle):
        '''Get the park state and maximum traveling distance of the vehicle
        Return: park state (1 for at home, 0 for not at home)
                predicted maximum travel distance
                cruiseBackHour: Boolean, Whether it is the hour vehicle returns to home, 
                    the tankVol is reduced at this hour
        '''
        weekday = self.timeIndex[self.time_step_idx].weekday()
        hour = self.timeIndex[self.time_step_idx].hour
        if weekday:
            vehicle_park = vehicle.parkSchd_wd[hour]
            cruiseHour = vehicle.parkSchd_wd.index[vehicle.parkSchd_wd==0].max()+1
            vehicle_max_dist = vehicle.dist_mu_wd+5*vehicle.dist_sigma_wd
        else:
            vehicle_park = vehicle.parkSchd_nwd[hour]
            cruiseHour = vehicle.parkSchd_wd.index[vehicle.parkSchd_wd==0].max()+1
            vehicle_max_dist = vehicle.dist_mu_nwd+5*vehicle.dist_sigma_nwd
        cruiseBackHour = hour == cruiseHour
        return vehicle_park, vehicle_max_dist, cruiseBackHour