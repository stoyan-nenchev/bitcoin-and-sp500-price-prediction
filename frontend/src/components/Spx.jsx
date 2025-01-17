import { useState, useEffect } from 'react';
import styled from "styled-components";
import { getHistoricalSpxData, getPredictedSpxData } from '../services/spxService';
import PriceChart from './PriceChart';

const Spx = () => {
    const [historicalData, setHistoricalData] = useState([]);
    const [predictedData, setPredictedData] = useState([]);

    useEffect(() => {
        const fetchHistoricalData = async () => {
            setHistoricalData(await getHistoricalSpxData());
        };
        const fetchPredictedData = async () => {
            setPredictedData(await getPredictedSpxData());
        };

        fetchHistoricalData();
        fetchPredictedData();
    }, []);

    return (
        <Container>
            <StyledTitle>Current S&P500 Price</StyledTitle>
            <PriceChart data={historicalData} domain={[0, 7500]} />
            <StyledTitle>Projected S&P500 Price</StyledTitle>
            <PriceChart data={predictedData} domain={[0, 10000]} />
        </Container>
    );
};

const Container = styled.div`
    padding: 40px;
`

const StyledTitle = styled.p`
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 20px;
`;

export default Spx;