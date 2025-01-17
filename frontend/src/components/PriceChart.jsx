import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from "recharts";
import dayjs from "dayjs";

const PriceChart = ({ data, domain }) => {
    return (
        <ResponsiveContainer width="100%" height={400}>
            <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tickFormatter={(tick) => dayjs(tick).format("YYYY")} />
                <YAxis domain={domain} />
                <Tooltip />
                <Line
                    type="monotone"
                    dataKey="price"
                    stroke="#8884d8"
                    dot={false}
                />
            </LineChart>
        </ResponsiveContainer>
    );
};

export default PriceChart;
