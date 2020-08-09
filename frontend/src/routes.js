import React from "react"
import { Route } from "react-router-dom";
import Home from './pages/Home';
import IndividualNetwork from './pages/IndividualNetwork';

const BaseRouter = () => {
    return(
        <div>
            <Route exact path='/' component={Home}/>
            <Route exact path='/networks/:networkID' component={IndividualNetwork}/>
        </div>
    )
}

export default BaseRouter;