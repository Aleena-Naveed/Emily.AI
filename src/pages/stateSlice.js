import { createSlice } from '@reduxjs/toolkit'

export const stateSlice = createSlice({
    name: 'states',
    initialState: {
        islogin: false,
        user: {},
        token: '',
        name: "",
        id: "",
        code: '',
    },
    reducers: {
        login: (state) => {
            state.islogin = true;
        },
        logout: (state) => {
            state.islogin = false;
        },
        settoken: (state, action) => {
            state.token = action.payload;
        },
        setuser: (state, action) => {
            state.user = action.payload;
            state.name = action.payload.name ? action.payload.name.split(' ')[0] : "";
            state.id = action.payload._id.slice(0, 10);
        },
        setcode: (state, action) => {
            state.codeoken = action.payload;
        },
    },
})

// Action creators are generated for each case reducer function
export const { login, logout, settoken, setuser, setcode } = stateSlice.actions

export default stateSlice.reducer