import { useState, useEffect } from 'react';
import styled from "styled-components";
import { getHistoricalBitcoinData, getPredictedBitcoinData } from '../services/bitcoinService';
import PriceChart from './PriceChart';

const Bitcoin = () => {
    const [historicalData, setHistoricalData] = useState([]);
    const [predictedData, setPredictedData] = useState([]);

    useEffect(() => {
        const fetchHistoricalData = async () => {
            setHistoricalData(await getHistoricalBitcoinData());
        };
        const fetchPredictedData = async () => {
            setPredictedData(await getPredictedBitcoinData());
        };

        fetchHistoricalData();
        fetchPredictedData();
    }, []);

    return (
        <Container>
            <StyledTitle>Current Bitcoin Price</StyledTitle>
            <PriceChart data={historicalData} domain={[0, 125000]} />
            <StyledTitle>Projected Bitcoin Price</StyledTitle>
            <PriceChart data={predictedData} domain={[0, 150000]} />
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

export default Bitcoin;
