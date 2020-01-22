# -*- coding: utf-8 -*-
"""
@author: amy-f
"""
import argparse

'''
################################################################################
'''


class Parser:
    """ Inits and parses the flags necessary to run the examples """

    ############################
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("start", nargs="+", type=float)
        self.parser.add_argument("target", nargs="+", type=float)
        self.parser.add_argument("maxJ", type=int)
        self.parser.add_argument("-l", "--load", help="load saved data", action="store_true")
        self.parser.add_argument("-s", "--save", help="save data as array", action="store_true")
        self.parser.add_argument("-g", "--gif", help="save final animation as gif", action="store_true")
        self.parser.add_argument("-p", "--plot", help="show cost2go dynamically", action="store_true")

    ##############################

    def parse(self, n_dim):
        args = self.parser.parse_args()
        if len(args.start) != n_dim:
            raise ValueError("Start point doesn't match nb dim")
        if len(args.target) != n_dim:
            raise ValueError("Target point doesn't match nb dim")
        return args

    #############################
