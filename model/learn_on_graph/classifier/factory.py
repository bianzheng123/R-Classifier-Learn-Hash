from model.learn_on_graph.classifier.datanode import use_neighbor, use_self


def create_datanode(config):
    # 这里的config指的是classifier_config
    config['data_prepare_config']['k'] = config['k']
    config['data_prepare_config']['n_cluster'] = config['n_cluster']
    if config['use_neighbor'] is True:
        return use_neighbor.UseNeighborDatanode(config['data_prepare_config'])
    elif config['use_neighbor'] is False:
        return use_neighbor.UseSelfDatanode(config['data_prepare_config'])
    else:
        raise Exception('找不到相关的datanode')
