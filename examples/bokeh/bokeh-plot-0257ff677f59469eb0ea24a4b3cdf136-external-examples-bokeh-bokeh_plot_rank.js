(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("214cb98a-9044-4a46-b6a3-774a62d1b542");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '214cb98a-9044-4a46-b6a3-774a62d1b542' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"fc654fbc-941a-48b1-8d35-0ec57454019b":{"roots":{"references":[{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"26842","type":"Span"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26915"},"selection_policy":{"id":"26914"}},"id":"26877","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"26859"},"glyph":{"id":"26860"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26861"},"selection_glyph":null,"view":{"id":"26863"}},"id":"26862","type":"GlyphRenderer"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26898"},"selection_policy":{"id":"26897"}},"id":"26843","type":"ColumnDataSource"},{"attributes":{"source":{"id":"26859"}},"id":"26863","type":"CDSView"},{"attributes":{"below":[{"id":"26770"}],"center":[{"id":"26773"},{"id":"26777"},{"id":"26836"},{"id":"26842"},{"id":"26848"},{"id":"26854"}],"left":[{"id":"26774"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26834"},{"id":"26840"},{"id":"26846"},{"id":"26852"}],"title":{"id":"26857"},"toolbar":{"id":"26788"},"toolbar_location":null,"x_range":{"id":"26762"},"x_scale":{"id":"26766"},"y_range":{"id":"26764"},"y_scale":{"id":"26768"}},"id":"26761","subtype":"Figure","type":"Plot"},{"attributes":{"data_source":{"id":"26849"},"glyph":{"id":"26850"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26851"},"selection_glyph":null,"view":{"id":"26853"}},"id":"26852","type":"GlyphRenderer"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"26848","type":"Span"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26896"},"selection_policy":{"id":"26895"}},"id":"26837","type":"ColumnDataSource"},{"attributes":{"source":{"id":"26837"}},"id":"26841","type":"CDSView"},{"attributes":{},"id":"26915","type":"Selection"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26786","type":"BoxAnnotation"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26833","type":"VBar"},{"attributes":{},"id":"26914","type":"UnionRenderers"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26838","type":"VBar"},{"attributes":{"ticks":[0,1,2,3]},"id":"26855","type":"FixedTicker"},{"attributes":{},"id":"26895","type":"UnionRenderers"},{"attributes":{"callback":null},"id":"26819","type":"HoverTool"},{"attributes":{"data_source":{"id":"26837"},"glyph":{"id":"26838"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26839"},"selection_glyph":null,"view":{"id":"26841"}},"id":"26840","type":"GlyphRenderer"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26839","type":"VBar"},{"attributes":{},"id":"26899","type":"UnionRenderers"},{"attributes":{},"id":"26783","type":"UndoTool"},{"attributes":{"overlay":{"id":"26787"}},"id":"26782","type":"LassoSelectTool"},{"attributes":{},"id":"26900","type":"Selection"},{"attributes":{},"id":"26800","type":"LinearScale"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"26864","type":"Span"},{"attributes":{},"id":"26784","type":"SaveTool"},{"attributes":{"axis":{"id":"26804"},"ticker":null},"id":"26807","type":"Grid"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26866","type":"VBar"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26832","type":"VBar"},{"attributes":{},"id":"26762","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26778"},{"id":"26779"},{"id":"26780"},{"id":"26781"},{"id":"26782"},{"id":"26783"},{"id":"26784"},{"id":"26785"}]},"id":"26788","type":"Toolbar"},{"attributes":{"text":"tau"},"id":"26857","type":"Title"},{"attributes":{},"id":"26897","type":"UnionRenderers"},{"attributes":{"ticks":[0,1,2,3]},"id":"26883","type":"FixedTicker"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26890"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26771"}},"id":"26770","type":"LinearAxis"},{"attributes":{},"id":"26905","type":"BasicTickFormatter"},{"attributes":{},"id":"26893","type":"UnionRenderers"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26888"},"ticker":{"id":"26855"}},"id":"26774","type":"LinearAxis"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26851","type":"VBar"},{"attributes":{"data_source":{"id":"26843"},"glyph":{"id":"26844"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26845"},"selection_glyph":null,"view":{"id":"26847"}},"id":"26846","type":"GlyphRenderer"},{"attributes":{},"id":"26768","type":"LinearScale"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26909"},"selection_policy":{"id":"26908"}},"id":"26859","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"26770"},"ticker":null},"id":"26773","type":"Grid"},{"attributes":{"callback":null},"id":"26785","type":"HoverTool"},{"attributes":{"source":{"id":"26849"}},"id":"26853","type":"CDSView"},{"attributes":{},"id":"26888","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"26843"}},"id":"26847","type":"CDSView"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26787","type":"PolyAnnotation"},{"attributes":{"data_source":{"id":"26831"},"glyph":{"id":"26832"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26833"},"selection_glyph":null,"view":{"id":"26835"}},"id":"26834","type":"GlyphRenderer"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"26854","type":"Span"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26911"},"selection_policy":{"id":"26910"}},"id":"26865","type":"ColumnDataSource"},{"attributes":{"children":[{"id":"26919"},{"id":"26917"}]},"id":"26920","type":"Column"},{"attributes":{},"id":"26903","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"26871"}},"id":"26875","type":"CDSView"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26867","type":"VBar"},{"attributes":{},"id":"26781","type":"WheelZoomTool"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26879","type":"VBar"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26844","type":"VBar"},{"attributes":{},"id":"26890","type":"BasicTickFormatter"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"26876","type":"Span"},{"attributes":{"below":[{"id":"26804"}],"center":[{"id":"26807"},{"id":"26811"},{"id":"26864"},{"id":"26870"},{"id":"26876"},{"id":"26882"}],"left":[{"id":"26808"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26862"},{"id":"26868"},{"id":"26874"},{"id":"26880"}],"title":{"id":"26885"},"toolbar":{"id":"26822"},"toolbar_location":null,"x_range":{"id":"26762"},"x_scale":{"id":"26800"},"y_range":{"id":"26764"},"y_scale":{"id":"26802"}},"id":"26797","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"26913","type":"Selection"},{"attributes":{"children":[[{"id":"26761"},0,0],[{"id":"26797"},0,1]]},"id":"26917","type":"GridBox"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"26836","type":"Span"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"26882","type":"Span"},{"attributes":{"data_source":{"id":"26877"},"glyph":{"id":"26878"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26879"},"selection_glyph":null,"view":{"id":"26881"}},"id":"26880","type":"GlyphRenderer"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26812"},{"id":"26813"},{"id":"26814"},{"id":"26815"},{"id":"26816"},{"id":"26817"},{"id":"26818"},{"id":"26819"}]},"id":"26822","type":"Toolbar"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26860","type":"VBar"},{"attributes":{},"id":"26896","type":"Selection"},{"attributes":{},"id":"26817","type":"UndoTool"},{"attributes":{},"id":"26766","type":"LinearScale"},{"attributes":{"overlay":{"id":"26821"}},"id":"26816","type":"LassoSelectTool"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26900"},"selection_policy":{"id":"26899"}},"id":"26849","type":"ColumnDataSource"},{"attributes":{},"id":"26764","type":"DataRange1d"},{"attributes":{},"id":"26815","type":"WheelZoomTool"},{"attributes":{},"id":"26894","type":"Selection"},{"attributes":{},"id":"26912","type":"UnionRenderers"},{"attributes":{"overlay":{"id":"26786"}},"id":"26780","type":"BoxZoomTool"},{"attributes":{},"id":"26818","type":"SaveTool"},{"attributes":{"source":{"id":"26865"}},"id":"26869","type":"CDSView"},{"attributes":{},"id":"26812","type":"ResetTool"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26873","type":"VBar"},{"attributes":{},"id":"26898","type":"Selection"},{"attributes":{},"id":"26813","type":"PanTool"},{"attributes":{},"id":"26911","type":"Selection"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26821","type":"PolyAnnotation"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26872","type":"VBar"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26878","type":"VBar"},{"attributes":{},"id":"26779","type":"PanTool"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"26870","type":"Span"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26894"},"selection_policy":{"id":"26893"}},"id":"26831","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"26820"}},"id":"26814","type":"BoxZoomTool"},{"attributes":{"axis":{"id":"26774"},"dimension":1,"ticker":null},"id":"26777","type":"Grid"},{"attributes":{"data_source":{"id":"26865"},"glyph":{"id":"26866"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26867"},"selection_glyph":null,"view":{"id":"26869"}},"id":"26868","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"26808"},"dimension":1,"ticker":null},"id":"26811","type":"Grid"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26913"},"selection_policy":{"id":"26912"}},"id":"26871","type":"ColumnDataSource"},{"attributes":{},"id":"26908","type":"UnionRenderers"},{"attributes":{"toolbar":{"id":"26918"},"toolbar_location":"above"},"id":"26919","type":"ToolbarBox"},{"attributes":{"data_source":{"id":"26871"},"glyph":{"id":"26872"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26873"},"selection_glyph":null,"view":{"id":"26875"}},"id":"26874","type":"GlyphRenderer"},{"attributes":{},"id":"26802","type":"LinearScale"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26850","type":"VBar"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26861","type":"VBar"},{"attributes":{},"id":"26778","type":"ResetTool"},{"attributes":{},"id":"26805","type":"BasicTicker"},{"attributes":{"toolbars":[{"id":"26788"},{"id":"26822"}],"tools":[{"id":"26778"},{"id":"26779"},{"id":"26780"},{"id":"26781"},{"id":"26782"},{"id":"26783"},{"id":"26784"},{"id":"26785"},{"id":"26812"},{"id":"26813"},{"id":"26814"},{"id":"26815"},{"id":"26816"},{"id":"26817"},{"id":"26818"},{"id":"26819"}]},"id":"26918","type":"ProxyToolbar"},{"attributes":{},"id":"26909","type":"Selection"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26905"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26805"}},"id":"26804","type":"LinearAxis"},{"attributes":{},"id":"26771","type":"BasicTicker"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26820","type":"BoxAnnotation"},{"attributes":{"source":{"id":"26877"}},"id":"26881","type":"CDSView"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26903"},"ticker":{"id":"26883"}},"id":"26808","type":"LinearAxis"},{"attributes":{"source":{"id":"26831"}},"id":"26835","type":"CDSView"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26845","type":"VBar"},{"attributes":{"text":"mu"},"id":"26885","type":"Title"},{"attributes":{},"id":"26910","type":"UnionRenderers"}],"root_ids":["26920"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"fc654fbc-941a-48b1-8d35-0ec57454019b","root_ids":["26920"],"roots":{"26920":"214cb98a-9044-4a46-b6a3-774a62d1b542"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();