import axiosInstance from "../axiosInstance";

export const getHistoricalSpxData = async () => {
    try {
        const { data } = await axiosInstance.get("spx_data");
        
        return data;
    } catch (error) {
        console.log(error);
    }
};

export const getPredictedSpxData = async () => {
    try {
        const { data } = await axiosInstance.get("predict_spx_price?days=1825");

        return data;
    } catch (error) {
        console.log(error);
    }
};