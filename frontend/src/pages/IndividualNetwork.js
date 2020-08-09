import React, { Component } from "react";
import axios from "axios";
import Network from "../components/Network";
import Layer from "../components/Layer";

class IndividualNetwork extends Component {
    constructor(props) {
        super(props);
        this.state = {
            network: {
                network_name: "default",
                owner: "default",
                number_of_layers: 0
            },
            layers: [],
            id: 0
        }
    }

    componentDidMount() {
        const id = this.props.match.params.networkID;
        this.setState({ id: id})
        axios.all([
            axios.get(`http://127.0.0.1:8000/api/network/${id}`),
            axios.get('http://127.0.0.1:8000/api/layer')
        ])
            .then(axios.spread((networkData, layersData) => {
                this.setState({
                    network: networkData.data,
                    layers: layersData.data
                });
                console.log(this.state.layers);
        }))
    }

    render() {
        return(
            <div>
                <p>This is our network!</p>
                <Network data={this.state.network}/>
                <p>And these are our layers!</p>
                {this.state.layers.map(x => <Layer data={x} key={x.id}/>)}
            </div>
        )
    }
}

export default IndividualNetwork;