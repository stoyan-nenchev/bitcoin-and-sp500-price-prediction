import styled from "styled-components";
import { useNavigate } from "react-router-dom";

const Header = () => {
    const navigate = useNavigate();

    return (
        <StyledHeader>
            <Wrapper>
                <ItemsContainer>
                    <Item onClick={() => navigate("/")}>S&P500</Item>
                    <Item onClick={() => navigate("/bitcoin")}>Bitcoin</Item>
                </ItemsContainer>
            </Wrapper>
        </StyledHeader>
    );
};

const StyledHeader = styled.div`
    display: flex;
    align-items: center;
    background-color: #111111;
    height: 80px;
    width: 100%;
`;

const Wrapper = styled.div`
    max-width: 60%;
    margin: 0 auto;
    padding: 20px 0;
`;

const ItemsContainer = styled.div`
    display: flex;
`

const Item = styled.div`
    color: #ffffff;
    font-size: 18px;
    font-weight: 600;
    margin: 0 20px;
    cursor: pointer;
`;

export default Header;
