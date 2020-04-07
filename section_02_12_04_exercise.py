#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:28:57 2020

@author: dustan
"""

class Employee(object):
    def __init__(self, name, years_of_service):
        self.name = name
        self.years_of_service = years_of_service
    def salary(self):
        return 1500+100*self.years_of_service

class Manager(Employee):
    def salary(self):
        return 2500 + 120*self.years_of_service

database = {}
employees = [Employee('lucy', 3), Employee('john', 1), Manager('julie', 10), Manager('paul', 3)]
for emp in employees:
    database[emp] = emp

table = [[name, database[name].salary()] for name in database.keys()]
print(table)
average_salary = sum(tt[1] for tt in table)/len(table)
print(average_salary)