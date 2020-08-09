import React, { Component } from "react"

class Layer extends Component {
    
    render() {
        return(
            <div>
                <p>This is a layer!</p>
                <p>{this.props.parent_network}</p>
                <p>{this.props.biases}</p>
                <p>{this.props.weights}</p>
            </div>
        );
    }
}

export default Layer;