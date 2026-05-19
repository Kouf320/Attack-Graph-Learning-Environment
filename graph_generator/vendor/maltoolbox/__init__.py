# -*- encoding: utf-8 -*-
# MAL Toolbox v0.0.21
# Copyright 2023, Andrei Buhaiu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""
MAL-Toolbox Framework
"""

__title__ = 'maltoolbox'
__version__ = '0.0.21'
__authors__ = ['Andrei Buhaiu']
__license__ = 'Apache 2.0'
__docformat__ = 'restructuredtext en'

__all__ = ()

import os
import sys
import tempfile
import configparser
import logging

ERROR_INCORRECT_CONFIG = 1

# Vendored-copy patch:
# Resolve the bundled config relative to THIS file rather than via installed
# distribution metadata (pkg_resources / egg-info).  This lets the package run
# as a fully self-contained drop-in inside graph_generator/vendor with no
# `pip install` of mal-toolbox required.
CONFIGFILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "default.conf"
)

config = configparser.ConfigParser()
config.read(CONFIGFILE)

if 'logging' not in config:
    print('Config file is missing essential information, cannot proceed.')
    sys.exit(ERROR_INCORRECT_CONFIG)

for term in ['output_dir', 'log_file']:
    if term not in config['logging']:
        logger.critical('Config file is missing essential '\
            'information, cannot proceed.')
        print('Config file is missing essential information, cannot '\
            'proceed.')
        sys.exit(ERROR_INCORRECT_CONFIG)

# Vendored-copy patch:
# default.conf uses a *relative* output_dir ("tmp"), which would create a tmp/
# folder in whatever directory the tool is launched from.  Redirect all log
# artefacts to the OS temp dir so running the generator never litters the repo.
_LOG_BASE = os.path.join(tempfile.gettempdir(), 'maltoolbox')
log_configs = {
    'output_dir': _LOG_BASE,
    'log_file': os.path.join(_LOG_BASE, 'log.txt'),
    'attackgraph_file': os.path.join(_LOG_BASE, 'attackgraph.json'),
    'model_file': os.path.join(_LOG_BASE, 'model.json'),
    'langspec_file': os.path.join(_LOG_BASE, 'langspec_file.json'),
}

os.makedirs(log_configs['output_dir'], exist_ok = True)
logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=log_configs["log_file"],
            filemode='w')
logging.getLogger('python_jsonschema_objects').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

if 'neo4j' in config:
    for term in ['uri', 'username', 'password', 'dbname']:
        if term not in config['neo4j']:
            logger.critical('Config file is missing essential '\
                f'Neo4J information: {term}, cannot proceed.')
            print('Config file is missing essential '\
                f'Neo4J information: {term}, cannot proceed.')
            sys.exit(ERROR_INCORRECT_CONFIG)

    neo4j_configs = {
        'uri': config['neo4j']['uri'],
        'username': config['neo4j']['username'],
        'password': config['neo4j']['password'],
        'dbname': config['neo4j']['dbname'],
    }

