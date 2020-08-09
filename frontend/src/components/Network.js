import React, { Component } from "react";

class Network extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        return(
            <div>
                <p>Network Name: {this.props.data.network_name}</p>
                <p>Created by: {this.props.data.owner}</p>
                <p>Number of Layers: {this.props.data.number_of_layers}</p>
            </div>
        )
    }
}

export default Network;