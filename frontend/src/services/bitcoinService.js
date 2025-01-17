import axiosInstance from "../axiosInstance";

export const getHistoricalBitcoinData = async () => {
    try {
        const { data } = await axiosInstance.get("bitcoin_data");
        
        return data;
    } catch (error) {
        console.log(error);
    }
};

export const getPredictedBitcoinData = async () => {
    try {
        const { data } = await axiosInstance.get("predict_bitcoin_price?days=1825");

        return data;
    } catch (error) {
        console.log(error);
    }
};